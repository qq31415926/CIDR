from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig, DistilBertModel, DistilBertConfig
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.distributions import Categorical
from captum.attr import Saliency
class bertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bertconfig = BertConfig.from_pretrained(config.bert_path)
        self.bertconfig.num_labels = config.num_classes

        self.bertmodel = BertModel.from_pretrained(config.bert_path, config=self.bertconfig)
        self.num_labels = config.num_classes
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        # for name,param in self.bertmodel.named_parameters():
        #     if "embedding" in name:
        #         param.requires_grad = False

    def forward_for_IG(self, input_ids, token_type_ids):
        x = self.bertmodel.embeddings(input_ids, token_type_ids)
        return x

    def forward(self, input_ids, return_pool=False, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        # print(input_ids.shape)
        outputs = self.bertmodel(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        pooled_emb = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                # print("loss:{} labels:{}".format(logits,labels))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # print("loss:")
            outputs = (loss,) + outputs
        if not return_pool:
            if labels is not None:
                return outputs  # (loss), logits, (hidden_states), (attentions)
            else:
                return outputs[0]
        else:
            return pooled_emb, pooled_output


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class robertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        roberta_config = RobertaConfig.from_pretrained(config.bert_path)
        roberta_config.num_labels = config.num_classes
        roberta_config.hidden_dropout_prob = config.hidden_dropout_prob
        self.roberta = RobertaModel.from_pretrained(config.bert_path, config=roberta_config)

        self.num_labels = config.num_classes
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        # self.classifier = RobertaClassificationHead(roberta_config)

    def forward(self, input_ids, return_pool=False, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        # pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # print(sequence_output.shape)
        # logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)

            if self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                loss_fct = CrossEntropyLoss()
                # print(logits.shape,labels.shape)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        output = (logits,) + outputs[2:]
        if not return_pool:
            return ((loss,) + output) if loss is not None else output
        else:
            return sequence_output, outputs[1]


class distilbertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        distilconfig = DistilBertConfig.from_pretrained(config.bert_path)
        distilconfig.num_labels = config.num_classes
        distilconfig.hidden_dropout_prob = config.hidden_dropout_prob
        self.distilbert_model = DistilBertModel.from_pretrained(config.bert_path, config=distilconfig)

        self.num_labels = config.num_classes
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.prelinear = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, return_pool=False, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        outputs = self.distilbert_model(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask
        )
        sequence_output = outputs[0]  # B, L, D

        # avg embedding
        # pooled_output = torch.mean(sequence_output, dim = 1) # B, D
        pooled_output = sequence_output[:, 0, :]
        # pooled_output = outputs[1]
        # pooled_output = self.prelinear(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # print(sequence_output.shape)
        # logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)

            if self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                loss_fct = CrossEntropyLoss()
                # print(logits.shape,labels.shape)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        output = (logits,) + outputs[2:]
        if not return_pool:
            return ((loss,) + output) if loss is not None else output
        else:
            return sequence_output, pooled_output


class LSTMEncoder(nn.Module):
    def __init__(self, args, vectors):

        super(LSTMEncoder, self).__init__()

        self.args = args
        # print(args.embed_dim)
        args.embed_dim = 300
        self.embed = nn.Embedding(args.embed_num, args.embed_dim, padding_idx=0)

        # initialize word embedding with pretrained word2vec
        self.embed.weight.data.copy_(torch.from_numpy(vectors))

        # fix embedding
        if args.mode == 'static':
            self.embed.weight.requires_grad = False
        else:
            self.embed.weight.requires_grad = True

        # <unk> vectors is randomly initialized
        # nn.init.uniform_(self.embed.weight.data[0], -0.05, 0.05)

        # <pad> vector is initialized as zero padding
        # nn.init.constant_(self.embed.weight.data[1], 0)

        # lstm
        self.lstm = nn.LSTM(args.embed_dim, args.lstm_hidden_dim, num_layers=args.lstm_hidden_layer)
        # initial weight
        init.xavier_normal_(self.lstm.all_weights[0][0], gain=np.sqrt(6.0))
        init.xavier_normal_(self.lstm.all_weights[0][1], gain=np.sqrt(6.0))

        # linear
        self.hidden2label = nn.Linear(args.lstm_hidden_dim, args.num_classes)
        # dropout
        # self.dropout = nn.Dropout(args.hidden_dropout_prob)
        # self.dropout_embed = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, x, training = True):

        # lstm
        pack_x, _ = self.lstm(x)
        # lstm_out = pack_x.data
        # print(hidden[0].shape,hidden[1].shape)
        # lstm_out = hidden[0,:,:] # B,H
        lstm_out,_ = pad_packed_sequence(pack_x,batch_first=True)
        # lstm_out B,T,D
        # lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2) # B,D,T
        # pooling
        lstm_out = torch.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2) # B,D
        lstm_out = torch.tanh(lstm_out)
        lstm_out = F.dropout(lstm_out, p=self.args.hidden_dropout_prob, training=training)
        # linear
        logit = self.hidden2label(lstm_out)
        out = F.softmax(logit, 1)
        return out


class policy_net(nn.Module):
    def __init__(self,input_dim1,hidden_dim,dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim1,hidden_dim)
        # self.linear2 = nn.Linear(input_dim2,hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        # if use_sent:
        #     self.linear3 = nn.Linear(2*hidden_dim,2)
        # else:
        self.linear3 = nn.Linear(hidden_dim,2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, grad_x, sent_x = None):
        # if sent_x != None:
        #     sent_x = self.linear1(sent_x) 
        # grad_x = self.linear2(grad_x)
        # if sent_x != None:
        #     x = torch.cat(sent_x,grad_x,dim=1)
        # else:
        #     x = grad_x
        grad_x = self.linear1(grad_x) # B,L,H
        x = self.dropout(x) 
        x = self.linear3(x) # B,L,2
        x = F.relu(x)
        return F.softmax(x,dim=1) 


class MASK_LSTM(nn.Module):

    def __init__(self, args, vectors, use_policy=False):
        super(MASK_LSTM, self).__init__()
        self.args = args
        self.embed_dim = args.embed_dim
        self.device = args.device

        self.max_sent_len = args.max_seq_length

        self.use_policy = use_policy
    # self.training = args.training
        self.policy = None if use_policy == False else policy_net()
        self.lstmmodel = LSTMEncoder(args, vectors)


    def forward(self, batch, seq_lens,training=True):
        # embedding
        # batch B,L
        # seq_lens B List
        embed = self.lstmmodel.embed(batch)
        embed = F.dropout(embed, p=self.args.hidden_dropout_prob, training=training)

        # policy choice
        ################
        # if self.use_policy:
        #     saliency = Saliency()

        ################

        pack_x = pack_padded_sequence(embed,lengths=seq_lens,batch_first=True,enforce_sorted=False)

        # x = embed.view(len(x), embed.size(1), -1)  # seqlen, bsz, embed-dim


    #
        # if self.use_policy:
        #     pass
        # else:
        p = self.lstmmodel(pack_x,training)

        # p = self.vmask.get_statistics_batch(x)
        # x_prime = self.vmask(x, p, flag)
        # output = self.lstmmodel(x_prime)

        # self.infor_loss = F.softmax(p,dim=2)[:,:,1:2].mean()
        # probs_pos = F.softmax(p,dim=2)[:,:,1]
        # probs_neg = F.softmax(p,dim=2)[:,:,0]
        # infor_loss = torch.mean(probs_pos * torch.log(probs_pos+1e-8) + probs_neg*torch.log(probs_neg+1e-8))

        return p