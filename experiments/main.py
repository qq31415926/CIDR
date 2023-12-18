from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,RobertaTokenizer,RobertaConfig,
                                  RobertaForSequenceClassification)
from bert_encoder import bertEncoder,robertaEncoder,distilbertEncoder,MASK_LSTM
# from tran
from transformers import BertTokenizer,DistilBertConfig,DistilBertTokenizer,DistilBertForSequenceClassification
from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (convert_examples_to_features,
                        output_modes, processors,convert_examples_to_features_lstm)
from utils import read_vocab
from nltk.tokenize import word_tokenize
# from bert_mask_model import *
import sys
# from bert_VMASK.encoder import bertEncoder
import pickle
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence
from captum.attr import Saliency
from torch.nn import KLDivLoss

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta':(RobertaConfig , RobertaForSequenceClassification , RobertaTokenizer),
    'distilbert':(DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}
with open("id2token.pickle","rb") as f:
    id2token = pickle.load(f)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def train(args, train_dataset, model, tokenizer,vocab=None):
    # eval_dataset,example_book = load_and_cache_examples(args, args.task_name, tokenizer, type='dev')
    # eval_loss, eval_acc = evaluate(args, model, eval_dataset,example_book,tokenizer)
    # exit(0)
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.model_type != "lstm":
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        optimizer = Adam(model.parameters(),lr = args.learning_rate)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    epochnum = 0
    best_val_acc = None
    beta = args.beta
    loss = CrossEntropyLoss() if args.model_type == "lstm" else None
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        logging_steps = 0
        preds = None
        out_label_ids = None
        epochnum += 1
        count, trn_model_loss = 0, 0
        for step, batch in enumerate(epoch_iterator):
            count += 1
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids,all_ids,all_lens)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3] if args.model_type != "lstm" else batch[2],
                      }
            # (all_input_ids, all_input_mask, all_label_ids,all_ids,all_lens)

            if args.model_type == "lstm":
                # lens = []
                lens = batch[4].detach().cpu().numpy().tolist()
                outputs = model(batch[0],lens)
            else:
                outputs = model(input_ids = batch[0],
                                attention_mask = batch[1],
                                token_type_ids = batch[2] if args.model_type in ['bert', 'xlnet'] else None ,
                                labels = batch[3])
            if args.model_type != "lstm":
                model_loss, logits = outputs[:2]
            else:
                logits = outputs
                model_loss = loss(logits,inputs['labels'])
                
            
            batch_loss = model_loss
            trn_model_loss += batch_loss.item()

            # if args.n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            if args.model_type != "lstm":
                scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            logging_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        tr_acc = (preds == out_label_ids).mean()

        # evaluate model
        # eval_dataset,example_book = load_and_cache_examples(args, args.task_name, tokenizer, type='dev',split = args.split_data,vocab = vocab)
        # if args.cal_AOPC:
        #     eval_loss, eval_acc,aopc = evaluate(args, model, eval_dataset,example_book,tokenizer)
        # else:
        # eval_loss,eval_acc = evaluate(args, model, eval_dataset,example_book,tokenizer)
        # if not best_val_acc or eval_acc > best_val_acc:
        #     if not os.path.exists(args.output_dir):
        #         os.makedirs(args.output_dir)
        #     with open(os.path.join(args.output_dir, args.savename), 'wb') as f:
        #         torch.save(model, f)
        #     if args.model_type != "lstm":
        #         tokenizer.save_pretrained(args.output_dir)
        #     if args.split_data:
        #         torch.save(args, os.path.join(args.output_dir, 'training_args_split.bin'))
        #     else:
        #         torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        #     best_val_acc = eval_acc

        tr_loss = trn_model_loss / count
        print('epoch {} | train_loss {:.6f} | train_acc {:.6f}'.format(epochnum,tr_loss,tr_acc))
        # print('epoch {} | train_loss {:.6f} | train_acc {:.6f} | dev_loss {:.6f} | dev_acc {:.6f}'.format(epochnum,
                                                                                                        #   tr_loss,
        if epochnum % 1 == 0:
           if beta > 0.01:
               beta -= 0.099

    return global_step, tr_loss



def saliency_guided_training(args, train_dataset, model, tokenizer, vocab=None):
    # eval_dataset,example_book = load_and_cache_examples(args, args.task_name, tokenizer, type='dev')
    # eval_loss, eval_acc = evaluate(args, model, eval_dataset,example_book,tokenizer)
    # exit(0)
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.model_type != "lstm":
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        optimizer = Adam(model.parameters(),lr = args.learning_rate)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    epochnum = 0
    best_val_acc = None
    beta = args.beta
    criterion = CrossEntropyLoss() if args.model_type == "lstm" else None
    criterion_KL = KLDivLoss(reduction="batchmean")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        logging_steps = 0
        preds = None
        out_label_ids = None
        epochnum += 1
        count, trn_model_loss = 0, 0
        for step, batch in enumerate(epoch_iterator):
            count += 1
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids,all_ids,all_lens)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3] if args.model_type != "lstm" else batch[2],
                      }
            # (all_input_ids, all_input_mask, all_label_ids,all_ids,all_lens)

            if args.model_type == "lstm":
                # lens = []
                lens = batch[4].detach().cpu().numpy().tolist()
                outputs = model(batch[0],lens)
            else:
                outputs = model(input_ids = batch[0],
                                attention_mask = batch[1],
                                token_type_ids = batch[2] if args.model_type in ['bert', 'xlnet'] else None ,
                                labels = batch[3])
            if args.model_type != "lstm":
                model_loss, logits = outputs[:2]
            else:
                logits = outputs
                model_loss = criterion(logits,inputs['labels'])
                # loss += model_loss
                if args.use_policy:
                    model.eval()
                    saliency = Saliency(model)
                    embed = model.lstmmodel.embed(batch[0])
                    # B,L
                    grads = saliency.attribute(embed,inputs['labels']).mean(2).detach().cpu().to(dtype=torch.float)
                    temp_grads = grads.numpy().tolist()
                    temp_grads = [[0] + temp_grads_ + [0] for temp_grads_ in temp_grads]
                    # B,L,3
                    grad_tensor = [[ [temp_grads[b][l-1], temp_grads[b][l], temp_grads[b][l+1]] for l in range(1, temp_grads[0] - 1)]for b in range(len(temp_grads))]

                    grad_tensor = torch.tensor(grad_tensor).to(dtype=torch.float)
                    p_mask = model.policy(grad_tensor) # B,L,2
                    _, mask = torch.max(p_mask, dim = 2) # B,L

                    x_mask = batch[0] * mask
                    lens = torch.sum(mask,dim=1).to(dtype=torch.int).detach().cpu().numpy().tolist()
                    p_mask = model(x_mask,lens)
                    model_loss += criterion_KL(p_mask, logits)


            batch_loss = model_loss
            trn_model_loss += batch_loss.item()

            # if args.n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            if args.model_type != "lstm":
                scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            logging_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        tr_acc = (preds == out_label_ids).mean()

        # evaluate model
        # eval_dataset,example_book = load_and_cache_examples(args, args.task_name, tokenizer, type='dev',split = args.split_data,vocab = vocab)
        # if args.cal_AOPC:
        #     eval_loss, eval_acc,aopc = evaluate(args, model, eval_dataset,example_book,tokenizer)
        # else:
        # eval_loss,eval_acc = evaluate(args, model, eval_dataset,example_book,tokenizer)
        # if not best_val_acc or eval_acc > best_val_acc:
        #     if not os.path.exists(args.output_dir):
        #         os.makedirs(args.output_dir)
        #     with open(os.path.join(args.output_dir, args.savename), 'wb') as f:
        #         torch.save(model, f)
        #     if args.model_type != "lstm":
        #         tokenizer.save_pretrained(args.output_dir)
        #     if args.split_data:
        #         torch.save(args, os.path.join(args.output_dir, 'training_args_split.bin'))
        #     else:
        #         torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        #     best_val_acc = eval_acc

        tr_loss = trn_model_loss / count
        print('epoch {} | train_loss {:.6f} | train_acc {:.6f}'.format(epochnum,tr_loss,tr_acc))
        # print('epoch {} | train_loss {:.6f} | train_acc {:.6f} | dev_loss {:.6f} | dev_acc {:.6f}'.format(epochnum,
                                                                                                        #   tr_loss,
        if epochnum % 1 == 0:
           if beta > 0.01:
               beta -= 0.099

    return global_step, tr_loss    
def evaluate(args, model, eval_dataset,example_book,tokenizer):
    # cal_IG(args,eval_dataset,model,example_book,tokenizer)
    # exit(0)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    loss = CrossEntropyLoss() if args.model_type == "lstm" else None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        # with torch.no_grad():
        inputs = {  'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                    'labels':         batch[3] if args.model_type != "lstm" else batch[2]}
        if args.model_type == "lstm":
            lens = batch[4].detach().cpu().numpy().tolist()
            with torch.no_grad():
                outputs = model(batch[0],lens,training=False)
        else:
            # print(batch[3])
            with torch.no_grad():
                outputs = model(input_ids = batch[0],
                                token_type_ids = batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                                attention_mask = batch[1],
                                labels = batch[3])
        # outputs = model(inputs, 'eval')
        if args.model_type != "lstm":
            tmp_eval_loss, logits = outputs[:2]
        else:
            # logits = outputs
            # print(inputs['labels'].dtype)
            # inputs['labels'] = inputs['labels'].to(torch.long)
            with torch.no_grad():
                logits = outputs
                tmp_eval_loss = loss(logits,inputs['labels'])
            if args.use_policy:
                saliency = Saliency(model)
                embed = model.lstmmodel.embed(batch[0])
                # B,L
                grads = saliency.attribute(embed,inputs['labels']).mean(2).detach().cpu().to(dtype=torch.float)
                with torch.no_grad():
                    temp_grads = grads.numpy().tolist()
                    temp_grads = [[0] + temp_grads_ + [0] for temp_grads_ in temp_grads]
                    # B,L,3
                    grad_tensor = [[ [temp_grads[b][l-1], temp_grads[b][l], temp_grads[b][l+1]] for l in range(1, temp_grads[0] - 1)]for b in range(len(temp_grads))]

                    grad_tensor = torch.tensor(grad_tensor).to(dtype=torch.float)
                    p_mask = model.policy(grad_tensor) # B,L,2
                    _, mask = torch.max(p_mask, dim = 2) # B,L

                    x_mask = batch[0] * mask
                    lens = torch.sum(mask,dim=1).to(dtype=torch.int).detach().cpu().numpy().tolist()
                    p_mask = model(x_mask,lens)
                    logits = p_mask
                    mask_loss = loss(p_mask,inputs['labels'])
                    tmp_eval_loss = mask_loss
            # model_loss += criterion_KL(p_mask, logits)

                
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    
    eval_acc = (preds == out_label_ids).mean()
    
    return eval_loss, eval_acc


def load_and_cache_examples(args, task, tokenizer, type,split = False,vocab=None):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file

    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    
    print(cached_features_file)
    reload = args.reload
    if os.path.exists(cached_features_file) and not reload:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        with open(os.path.join(args.data_dir,"example_book.pickle"),"rb") as f:
            example_book = pickle.load(f)
    else:
        print("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        example_book = {}
        for example in examples:
            example_book[example.id] = example.text_a
        # print("example length:{}".format(len(examples)))
        # if split:
        #     print("raw dataset len:{}".format(len(examples)))
        #     choice_classes = ["0", "1"] 
        #     examples = filter(lambda e : e.label in choice_classes , examples)
            # print("split dataset len:{}".format(len(examples)))
            # split_example = []
            # choice_classes = ["0", "1"] 
            # for example in examples:
            #     if example.label in choice_classes:
            #         split_example.append(example)
            
        if args.model_type == "lstm":
            features = convert_examples_to_features_lstm(examples,label_list,args.max_seq_length,vocab)
        else:
            features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.model_type in ['roberta']),
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                    )
        print("split dataset len:{}".format(len(features)))
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        with open(os.path.join(args.data_dir,"example_book.pickle"),"wb") as f:
            pickle.dump(example_book,f)
        print("Saving example book into cached file %s", os.path.join(args.data_dir,"example_book.pickle"))


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)

    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long) if args.model_type != 'lstm' else None
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_ids = torch.tensor([f.id for f in features],dtype=torch.int)
    if args.model_type == "lstm":
        all_lens = torch.tensor([f.lens for f in features],dtype=torch.int)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids,all_ids,all_lens)

    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_ids)

    return dataset,example_book

def init_model(args):
    model_class, tokenizer_class = None, None
    if args.model_type != "lstm":
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # Load pretrained bert model
        bert_path = args.bert_path
        tokenizer_path = args.tokenizer_path
        # prebert = bertEncoder(args)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)
        vocab,vocab_list = None,None
        if args.model_type == "bert":
            model = bertEncoder(args)
        elif args.model_type == "roberta":
            model = robertaEncoder(args)
        elif args.model_type == "distilbert":
            model = distilbertEncoder(args)
    elif args.model_type == "lstm":
        tokenizer = None
        vocab,vocab_list = read_vocab(args.static_word_path)
        model = MASK_LSTM(args,vocab_list)
    return model,model_class,tokenizer_class,vocab,vocab_list
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='./dataset/data_name', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bert', type=str,
                        )
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default='SST-2', type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default='./output', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--savename', type=str, default='maskbert.pt',
                        help='path to save the final model')

    ## Other parameters
    parser.add_argument('-beta', type=float, default=1, help='beta')
    parser.add_argument('-mask-hidden-dim', type=int, default=100, help='number of hidden dimension')
    parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
            non-linearity transfer function')
    parser.add_argument('-embed-dim', type=int, default=768, help='original number of embedding dimension')
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=25, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
    parser.add_argument("--num_classes",default=2,type=int,help="number of classes")
    parser.add_argument("--reload",action="store_true")
    parser.add_argument("--log_dir",type=str,default="./logs")
    parser.add_argument("--bert_path",type=str,default=None)
    parser.add_argument("--tokenizer_path",type=str,default=None)
    parser.add_argument("--hidden_dropout_prob",type=float,default=0.1)
    parser.add_argument("--hidden_size",type=int,default=768)
    parser.add_argument("--use_IG",action="store_true")
    parser.add_argument("--use_VMASK",action="store_true")
    parser.add_argument("--cal_AOPC",action="store_true")
    parser.add_argument("--use_lime",action="store_true")
    parser.add_argument("--split_data",action="store_true")
    parser.add_argument("--static_word_path",default=None)
    parser.add_argument("--use_policy",action="store_true")
    parser.add_argument("--lstm_hidden_layer",type=int,default=1)
    parser.add_argument("--embed_dim",type=int,default=100)
    parser.add_argument("--lstm_hidden_dim",type=int,default=100)
    parser.add_argument("--embed_num",type=int,default=400000 + 2)
    parser.add_argument("--mode",type=str,default="static")
    parser.add_argument('--gamma', type=float)
    parser.add_argument("--lambda1",type=float)
    parser.add_argument("--lambda2",type=float)

    parser.add_argument("--input_dim1",type=int,default=3)
    parser.add_argument("--hidden_dim",type=int,default=10)
    parser.add_argument("--dropout",type=float,default=0.2)
    
    # parser.add_argument("--split_data")

    args = parser.parse_args()
    
    # if not os.path.exists("")
    # log_path = os.path.join(args.log_dir,args.task_name)
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(log_path,"output.log"))],
    # )
    # logger.setLevel(logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("cuda device:{}".format(args.gpu_id))
    if args.gpu > -1:
        args.device = "cuda"
    else:
        args.device = "cpu"
    args.n_gpu = 1

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    # label_list = processor.get_labels()
    args.model_type = args.model_type.lower()
    # print(args)
    if args.model_type != "lstm":
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        # Load pretrained bert model
        bert_path = args.bert_path
        tokenizer_path = args.tokenizer_path
        # prebert = bertEncoder(args)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)
        vocab = None
    #     if args.model_type == "bert":
    #         model = bertEncoder(args)
    #     elif args.model_type == "roberta":
    #         model = robertaEncoder(args)
    #     elif args.model_type == "distilbert":
    #         model = distilbertEncoder(args)
    elif args.model_type == "lstm":
        tokenizer = None
        vocab,vocab_list = read_vocab(args.static_word_path)
    #     model = MASK_LSTM(args,vocab_list)
    # print(model)
    # fix embeddings
    # parameters = filter(lambda p: p.requires_grad, model.bertmodel.bert.embeddings.parameters())
    # for param in parameters:
    #     param.requires_grad = False
    # for name, param in model.named_parameters():
    #     if "word_embeddings" in name:
    #         param.requires_grad = False

    # model,model_class,tokenizer_class,vocab,vocab_list = init_model(args)

    # del model
    # Training
    if args.split_data:
        train_dataset,_ = load_and_cache_examples(args, args.task_name, tokenizer, type='train',split=True)
    else:
        # if args.model_type != "lstm":
        train_dataset,_ = load_and_cache_examples(args, args.task_name, tokenizer, type='train',vocab=vocab)
        # else:
            # train_dataset,_ = load_and_cache_examples()

    print("Loaded data!")
    lrs = [1e-3] # 0.0001, 0.0005, 0.005, 0.001
    epochs = [40]
    hidden_dims = [200] # 100 300 500 
    batch_sizes = [64] # 64
    #     parser.add_argument("--input_dim1",type=int,default=5)
    # parser.add_argument("--hidden_dim",type=int,default=10)
    # parser.add_argument("--dropout",type=float,default=0.2)
    # input_dim1 = [5,10,15,20]
    hidden_dim_policy = [10,20,30]
    dropout = [0.1,0.2,0.3]
    # dropout 0.0 0.2 0.3

    best_val_acc = 0
    for hidden_dim in hidden_dim_policy:
        for dropout_p in dropout:
            args.hidden_dim = hidden_dim
            args.dropout = dropout_p
            model,model_class,tokenizer_class,vocab,vocab_list = init_model(args)
            model.to(args.device)
            if args.use_policy:
                global_step, tr_loss = train(args, train_dataset, model, tokenizer,vocab)
            else:
                global_step, tr_loss = saliency_guided_training(args, train_dataset, model, tokenizer, vocab=None)

            # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            # logger.info(" Training Finished!")
            print(" global_step = {}, average loss = {}".format(global_step, tr_loss))
            print(" Training Finished!")
            # Load the well-trained model and vocabulary that you have fine-tuned
            # del model
            # if args.model_type != "lstm":
            #     model = model_class.from_pretrained(args.output_dir)
            # with open(os.path.join(args.output_dir, args.savename), 'rb') as f:
            #     model = torch.load(f)
            # model.to(torch.device(args.device))
            if args.model_type != "lstm":
                tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

            # # Test
            test_dataset,example_book = load_and_cache_examples(args, args.task_name, tokenizer, type='test',vocab=vocab)
            test_loss,test_acc = evaluate(args, model, test_dataset,example_book,tokenizer)
            print("hidden_dim_policy:{} dropout:{} test_loss:{:.6f} test_acc:{:.6f}".format(hidden_dim_policy,dropout,test_loss,test_acc))
            # print('learning rate:{:.4f} epochs:{:.1f} hidden_dim:{:.1f} batch_size:{:.1f} test_loss {:.6f} | test_acc {:.6f}'.format(lr_,epoch,hidden_dim,batch_size,test_loss, test_acc))
            if not best_val_acc or test_acc > best_val_acc:
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                with open(os.path.join(args.output_dir, args.savename), 'wb') as f:
                    torch.save(model, f)
                if args.model_type != "lstm":
                    tokenizer.save_pretrained(args.output_dir)
                if args.split_data:
                    torch.save(args, os.path.join(args.output_dir, 'training_args_split.bin'))
                else:
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                best_val_acc = test_acc
            del model
            
    # for lr_ in lrs:
    #     for epoch in epochs:
    #         for hidden_dim in hidden_dims:
    #             for batch_size in batch_sizes:
    #                 args.learning_rate = lr_
    #                 args.num_train_epochs = epoch
    #                 args.per_gpu_train_batch_size = batch_size
    #                 args.lstm_hidden_dim = hidden_dim
    #                 model,model_class,tokenizer_class,vocab,vocab_list = init_model(args)
    #                 model.to(args.device)
    #                 global_step, tr_loss = train(args, train_dataset, model, tokenizer,vocab)
    #                 # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    #                 # logger.info(" Training Finished!")
    #                 print(" global_step = {}, average loss = {}".format(global_step, tr_loss))
    #                 print(" Training Finished!")
    #                 # Load the well-trained model and vocabulary that you have fine-tuned
    #                 # del model
    #                 # if args.model_type != "lstm":
    #                 #     model = model_class.from_pretrained(args.output_dir)
    #                 # with open(os.path.join(args.output_dir, args.savename), 'rb') as f:
    #                 #     model = torch.load(f)
    #                 # model.to(torch.device(args.device))
    #                 if args.model_type != "lstm":
    #                     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    #                 # # Test
    #                 test_dataset,example_book = load_and_cache_examples(args, args.task_name, tokenizer, type='test',vocab=vocab)
    #                 test_loss,test_acc = evaluate(args, model, test_dataset,example_book,tokenizer)
    #                 print('learning rate:{:.4f} epochs:{:.1f} hidden_dim:{:.1f} batch_size:{:.1f} test_loss {:.6f} | test_acc {:.6f}'.format(lr_,epoch,hidden_dim,batch_size,test_loss, test_acc))
    #                 if not best_val_acc or test_acc > best_val_acc:
    #                     if not os.path.exists(args.output_dir):
    #                         os.makedirs(args.output_dir)
    #                     with open(os.path.join(args.output_dir, args.savename), 'wb') as f:
    #                         torch.save(model, f)
    #                     if args.model_type != "lstm":
    #                         tokenizer.save_pretrained(args.output_dir)
    #                     if args.split_data:
    #                         torch.save(args, os.path.join(args.output_dir, 'training_args_split.bin'))
    #                     else:
    #                         torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    #                     best_val_acc = test_acc
    #                 del model
    # test_loss, test_acc,aopc = evaluate(args, model, test_dataset,example_book,tokenizer)
    # logger.info('\ntest_loss {:.6f} | test_acc {:.6f} aopc:{:.6f}'.format(test_loss, test_acc,aopc))
    # return test_loss, test_acc


if __name__ == "__main__":
    main()
