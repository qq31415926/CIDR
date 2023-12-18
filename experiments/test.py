from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import time
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification,RobertaTokenizer)
from bert_encoder import *
# from tran
from transformers import BertTokenizer,DistilBertConfig,DistilBertTokenizer,DistilBertForSequenceClassification
from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (convert_examples_to_features,
                        output_modes, processors)
import math
# from lime.lime_text import LimeTextExplainer
import sys
# from captum.attr import IntegratedGradients
from dp import cal_shapley_value,integrated_gradients,print_word_ig,minimum_feature_set,knapsack,calculate_IG,calculate_DIG_second,calculate_metric,calculate_DIG_first,calculate_CIG_norefine
# from bert_encoder import bertEncoder,robertaEn
import pickle
# import math
import numpy as np
from copy import deepcopy
from itertools import combinations
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta' : (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
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


# def soft_max(x):
    

def toks_score_to_word_score(toks,words,score):
    # print(toks)
    # print(words)
    # print(score)
    assert len(toks) == len(score)
    if len(toks) == len(words):
        return score
    
    score_w = []
    p = 0
    q = 0
    while p < len(toks):
        tok_p = toks[p]
        if tok_p[0] == "#":
            s_word = score[q]
            # print(s_word)
            score_w.pop()
            while p < len(toks) and toks[p][0] == "#":
                s_word = max(s_word,score[p])
                p += 1
            score_w.append(s_word)
            if p < len(toks):
                q = p
                # p += 1
        else:
            score_w.append(score[p])
            q = p
            p += 1
    # print(len(score_w))
    # print(len(words))
    assert len(score_w) == len(words)
    return score_w

def result_to_word(result,text,tok_score):
    '''
    result:[token]
    text:str
    tok_score[token score]
    '''
    # print(result)
    # print(text)
    text = text.split()
    # res = set()
    score = {}

    # for tok in result:
    for i in range(len(result)):
        tok = result[i]
        if tok[0] != "#":
            while tok[0] == "#":
                tok = tok[1:]
        for word in text:
            if tok in word:
                # res.add(word)
                if score.get(word,-1) == -1:
                    score[word] = tok_score[i]
                else:
                    score[word] = max(tok_score[i],score[word])
    return score
        # for word in text:

def pading_token(model_type,pad_type):
    pad_token = None
    if model_type == "bert" or model_type == "distilbert":
        if pad_type == "pad":
            pad_token = "[PAD]"
        elif pad_type == "mask":
            pad_token = "[MASK]"
        elif pad_type == "unk":
            pad_token = "[UNK]"
        else:
            pad_token = "del"
    elif model_type == "roberta":
        if pad_type == "pad":
            pad_token = "<pad>"
        elif pad_type == "mask":
            pad_token = "<mask>"
        elif pad_type == "unk":
            pad_token = "<unk>"
        else:
            pad_token = "del"
    return pad_token
def pading_text(text,model_type,max_L,add_token = True):
    pad_text = []
    mask = []

    for i in range(len(text)):
        l = len(text[i])
        if add_token:
            if model_type == "bert" or model_type == "distilbert":
                if  l  >= max_L - 2:
                    text[i] = text[i][:max_L - 2]
                    text[i] = ["[CLS]"] + text[i] + ["[SEP]"]
                    maski = [1] * (max_L)
                else:
                    text[i] = ["[CLS]"] + text[i] + ["[SEP]"] 
                    maski = [1] * len(text[i])
                    text[i] += ["[PAD]"]*(max_L  - len(text[i]))
                    maski += [0] * (max_L - 2 - l)
            elif model_type == "roberta":
                if l >= max_L - 3:
                    text[i] = text[i][:max_L - 3]
                    text[i] = ["<s>"] + text[i] + ["</s>"] * 2
                    maski = [1] * (max_L)    
                else:
                    text[i] = ["<s>"] + text[i] + ["</s>"] * 2
                    maski = [1] * len(text[i])
                    text[i] += ["<pad>"]*(max_L  - len(text[i]))
                    maski += [0] * (max_L - 3 - l)
        else:
            if model_type == "bert" or model_type == "distilbert":
                if  l  >= max_L:
                    text[i] = text[i][:max_L]
                    text[i] = text[i]
                    maski = [1] * (max_L)
                else:
                    maski = [1] * len(text[i])
                    text[i] += ["[PAD]"]*(max_L  - len(text[i]))
                    maski += [0] * (max_L  - l)
            elif model_type == "roberta":
                if l >= max_L :
                    text[i] = text[i][:max_L]
                    text[i] =  text[i] 
                    maski = [1] * (max_L)    
                else:
                    text[i] =  text[i] 
                    maski = [1] * len(text[i])
                    text[i] += ["<pad>"]*(max_L  - len(text[i]))
                    maski += [0] * (max_L  - l)
        pad_text.append(text[i]) 
        mask.append(maski)
    return pad_text,mask

                
def cal_AOPC(args,eval_dataset,model,example_book,tokenizer):
    # calculating AOPC 
    softmax = nn.Softmax(dim=-1)
    sigmoid = nn.Sigmoid()
    def predictor(texts,masks,is_tokenize=True):
        if not is_tokenize:
            
            if args.use_cls:
                if args.model_type != "roberta":
                    if "[PAD]" in texts:
                        pad_idx = texts.index("[PAD]")
                    else:
                        pad_idx = len(texts) - 1
                    texts = ["[CLS]"]+texts[:pad_idx] + ["[SEP]"] + texts[pad_idx:]
                    masks = [1] + masks[:pad_idx] + [1] + masks[pad_idx:]
                else:
                    if "<pad>" in texts:
                        pad_idx = texts.index("<pad>")
                    else:
                        pad_idx = len(texts) - 1
                    texts = ["<s>"]+texts[:pad_idx] + ["</s>","</s>"] + texts[pad_idx:]
                    masks = [1] + masks[:pad_idx] + [1,1] + masks[pad_idx:]    
                texts = [tokenizer.convert_tokens_to_ids(texts)]          
                # texts = [[101] + tokenizer.convert_tokens_to_ids(texts)]
                # masks = [1] + masks 
            else:
                texts = [tokenizer.convert_tokens_to_ids(texts)]

            inputs = {
                "input_ids":torch.IntTensor(texts).cuda(),
                "attention_masks":torch.IntTensor(masks).unsqueeze(0).cuda(),
            }
        else:
            
            # raw  = tokenizer(texts, return_tensors="pt", padding=True)
            texts = [tokenizer.convert_tokens_to_ids(t) for t in texts]
            input_ids = torch.IntTensor(texts).cuda()
            # print(input_ids.shape)
            mask = torch.IntTensor(masks).cuda()
            # print(mask.shape)
            inputs = {'input_ids':     input_ids, 
                      "attention_masks": mask
                                    }
        if args.use_VMASK:
            # print(model)
            outputs = model.forward_forlime(inputs)
            tensor_logits = outputs[0]

        else:
            if is_tokenize:
                outputs = model(input_ids = inputs["input_ids"] , attention_mask = inputs["attention_masks"])
            else:
                # print(inputs["input_ids"].shape)
                # print(inputs["attention_masks"].shape)
                outputs = model(input_ids = inputs["input_ids"] , attention_mask = inputs["attention_masks"])
            if type(outputs) == tuple:
                
                tensor_logits = outputs[0]
            else:
                tensor_logits = outputs

        # print(tensor_logits.shape)
        # if len(tensor_logits.shape) != 2:
        #     tensor_logits = tensor_logits.unsqueeze(0)
        probas = softmax(tensor_logits).cpu().detach().numpy()
        return probas
    eval_sampler = SequentialSampler(eval_dataset)
    
    bsz = args.per_gpu_eval_batch_size
    if args.useDIG:
        threshold = args.threshold
        lr = args.step_lr
    eval_dataloader = DataLoader(eval_dataset,sampler=eval_sampler,batch_size=bsz,)
    max_L = args.max_seq_length 
    threshold = args.threshold
    lr1 = args.step_lr1
    lr2 = args.step_lr2
    cnt = 0
    cnt_small = 0
    if args.cal_acc:
        delta_token = []
    aopc_token = pading_token(model_type = args.model_type,
                             pad_type = args.pad_token)
    # pad_token = "[PAD]" if args.model_type == "bert" or args.model_type == "distilbert" else "<pad>"
    if args.model_type == "bert" or args.model_type == "distilbert":
        if args.pad_token == "pad" or args.pad_token == "del":
            pad_token = "[PAD]"
        elif args.pad_token == "mask":
            pad_token = "[MASK]"
        elif args.pad_token == "unk":
            pad_token = "[UNK]"
    elif args.model_type == "roberta":
        if args.pad_token == "pad" or args.pad_token == "del":
            pad_token = "<pad>"
        elif args.pad_token == "mask":
            pad_token = "<mask>"
        elif args.pad_token == "unk":
            pad_token = "<unk>"  

    # if args.useDIG:
    #     res1,res2 = [] , []
        
    # else:
    #     res = []
    aopc_res,suff_res,log_res = [], [], []
    feature_essence_amount, feature_minimality_amount = 0,0
    for batch in tqdm(eval_dataloader,desc="Calculating AOPC"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        
        
        
        # Dynamic Integrated Gradients
        input_id = [batch[4][i].item() for i in range(len(batch[4]))]
        raw_text = [example_book[i] for i in input_id] # [str1, str2 ,str3,...,]
        # text = [t[:max_L - 2] for t in text]
        # text = [t.split()[:max_L - 2] for t in text]
        text = [tokenizer.tokenize(t) for t in raw_text]
        real_len = [len(t) for t in text]
        attn_mask = None
        text, attn_mask = pading_text(text = text, model_type = args.model_type,max_L = max_L)
            
        # print(len(text))
        with torch.no_grad():
            probs = predictor(text, attn_mask)
            # print(probs.shape)
        
        if args.useDIG or args.use_IG:
            if args.is_second:
                start_time = time.time()
                # b,l,l-1
                text_one_token_delete = [[inner_text[0:pick_idx] + inner_text[pick_idx + 1:] for pick_idx in range(len(inner_text))] for inner_text in text]
                # text_one_token_delete = 
                text_one_token_delete_mask = [[inner_mask[0:pick_idx] + inner_mask[pick_idx + 1:] for pick_idx in range(len(inner_mask))] for inner_mask in attn_mask ]
                text_one_token_delete = [[tokenizer.convert_tokens_to_ids(t) for t in t_] for t_ in text_one_token_delete]
                word_ig,grad_table = integrated_gradients(model=model,text_token=batch[0],token_mask=batch[1],y=batch[3],model_type=args.model_type,is_second=True,attr_token = text_one_token_delete,attr_mask = text_one_token_delete_mask) # B,N
                end_time = time.time()
                print("calculating second order ig time:{:.3f}min".format((end_time - start_time)/60))
            else:
                word_ig = integrated_gradients(model=model,text_token=batch[0],token_mask=batch[1],y=batch[3],model_type=args.model_type) # B,N

            # print(word_ig)
            # if args.model_type == "bert" or args.model_type == 'distilbert':
                # word_ig = word_ig[:,1:]
            # else:
            # remove cls token or <s> token
            word_ig = word_ig[:,1:]
            # word_ig = torch.sigmoid(word_ig)
            if args.useDIG:
                if not args.is_second:

                    with torch.no_grad():
                        word_mask = torch.gt(word_ig,0.0).int() # B,N
                        # index_pos = [[j for j in range(len(word_mask[b])) if word_mask[b][j].item() == 1] for b in range(len(word_mask))]
                        # print("word mask:{}".format(word_mask))
                        sum_of_ig = torch.sum(word_ig,dim=1) # B 
                        # print("sum of ig:{}".format(sum_of_ig))
                        word_pos_b = word_ig * word_mask

                        # pos_ig = torch.index_select(word_ig, dim = 1)
                        sum_pos = torch.sum(word_pos_b, dim = -1)

                        upper_bound = sum_pos - sum_of_ig + threshold

                        lower_bound = sum_pos - sum_of_ig  
                else:

                    # second order integrated gradients
                    word_mask = torch.gt(word_ig,0.0).int() # B,L - 1
                    # B,unknown , index belongs to [0,L - 1]
                    index_pos = [[j for j in range(len(word_mask[b]) - 1) if word_mask[b][j].item() == 1] for b in range(len(word_mask))]                    
                    
                    
                    upper_bound_p1 = [sum([grad_table[b][conbined_pair_a + 1][conbined_pair_b + 1].item() \
                        for conbined_pair_a,conbined_pair_b in combinations(index_pos[b],2)]) \
                            for b in range(len(word_mask))]
                    

                    index_all = [[piv for piv in range(word_mask.shape[1] - 1)] for _ in range(len(word_mask))] # B,L - 2
                    # combined_pairs_all = [combinations(index_pos_i,2) for index_pos_i in index_all] # B,(L - 2)*(L - 3) / 2
                    
                    second_order_ig_all =  [[[grad_table[b][bb + 1][bbb + 1].item() for bbb in index_all[b] ] for bb in index_all[b] ] for b in range(len(word_mask))]
                    # else:
                    #     second_order_ig_all =  [[[grad_table[b][bb + 2][bbb + 2].item() for bbb in index_all[b] ] for bb in index_all[b] ] for b in range(len(word_mask))]
                    upper_bound_p2 = [sum([second_order_ig_all[b][combined_pair_a][combined_pair_b] for combined_pair_a,combined_pair_b in combinations(index_all[b],2)]) for b in range(len(word_mask))]
                    upper_bound = [abs(-u_p1 + u_p2) for u_p1,u_p2 in zip(upper_bound_p1,upper_bound_p2)]
                    print("upper bound part1:{} upper bound part2:{} upper bound:{}".format(upper_bound_p1,upper_bound_p2,upper_bound))

            # else:
            #     word_ig = word_ig.detach().cpu().numpy().tolist()
                # att_score = att_score.detach().cpu().numpy().tolist()
            # print(text)
            
            toks = [ tokenizer.tokenize(t) for t in raw_text]
            # sum_l += 
            if args.model_type != "roberta":
                toks , _  = pading_text(text = toks,model_type = args.model_type,max_L = max_L - 2,add_token = False)
            else:
                toks , _  = pading_text(text = toks,model_type = args.model_type,max_L = max_L - 3,add_token = False)

            # print(toks)
            

                    
        # ans = [(token1,score1),(token2,score2),...,(token_n,score_n)]
        k = args.k
        
        if bsz > 1:
            # if args.useIG:
            #     cnt += 1
            print('')
            for i in range(min(bsz,len(probs))):
                
                label_id = int(np.argmax(probs[i,:],axis=-1))
                # label_id = batch[3][i].item()
                if args.use_IG:
            
                    aopc_i,suff_i,logodd_i,flag_ms1,max_aopc = calculate_IG(predictor,args,word_ig,batch,toks,aopc_token,pad_token,probs,i,label_id,k)
                    
                    aopc_i2 = 0
                    suff_i2 = 1
                    logodd_i2 = 0
                    # res.append(aopc_i)
                    # if len(res) >= args.evaluate_amount:
                    #     break
                elif args.useDIG:
                    # if len(res2) >= args.evaluate_amount:
                    #     break
                    input_id = batch[4][i].item()
                    word_ig_i = word_ig[i,:] # N
                    word_mask_i = word_mask[i,:] # N
                    # 
                    # print("word_ig_i:{}".format(word_ig_i))
                    if args.model_type != "roberta":
                        pos_idx = [ idx for idx in range(len(word_mask_i) - 1) if word_mask_i[idx].item() == 1 ]
                    else:
                        pos_idx = [ idx for idx in range(len(word_mask_i) - 2) if word_mask_i[idx].item() == 1 ]

                    
                    if len(pos_idx) == 0:

                        aopc_i,suff_i,logodd_i,flag_ms1,max_aopc = calculate_IG(predictor,args,word_ig,batch,toks,aopc_token,pad_token,probs,i,label_id,k)
                        
                        
                        aopc_i2 = 0
                        suff_i2 = 1
                        logodd_i2 = 0
                        
                    else:
                        if not args.is_second:
                            index_1 = torch.IntTensor(pos_idx).cuda()
                            word_pos = torch.index_select(word_ig_i, dim = 0 ,index = index_1) # postive feature
                            # def calculate_DIG_first(args,pos_idx, word_ig_i, word_pos, lower_bound, i, upper_bound,toks,):
                            ans, ans2 = calculate_DIG_first(args,pos_idx, word_ig_i, word_pos, lower_bound, i, upper_bound,toks)
                            del index_1
                            if len(ans) == 0:
                                continue
                            cnt += 1
                            if cnt > args.evaluate_amount:
                                break
                            ans = ans[:k]
                            print("dynamic feature set:{}".format(ans))
                            # print("Set1:{} Set2:{}".format(ans, ans2))
                            
                            aopc2 = []
                            cnt_tok = 0
                            if args.model_type == 'bert' or args.model_type == "distilbert":
                                cnt_tok = sum([1 if tok_ != "[PAD]" else 0 for tok_ in toks[i]])
                            else:
                                cnt_tok = sum([1 if tok_ != "<pad>" else 0 for tok_ in toks[i]])
                            s_text = deepcopy(toks[i])
                            print("cnt:{} raw token:{}".format(cnt_tok, s_text))
                            delta1 = []
                            print("ground truth:{} pred:{} prob:{}".format(batch[3][i].item(),label_id,probs[i,label_id].item()))
                            cnt_tok1 = cnt_tok
                            # if 
                            aopc_delta1,suff_delta1,logodd_delta1,flag_ms1 = calculate_metric(predictor,batch,i,label_id,probs,cnt_tok,ans,aopc_token,pad_token,s_text,metric="AOPC",second=False)
                            
                            # else:
                            # suff_delta1 = sorted(suff_delta1)
                            # suff_delta1 = suff_delta1[:len(suff_delta1) * 0.2]

                            aopc_i = np.array(aopc_delta1).mean() if len(aopc_delta1) != 0 else 0
                            suff_i = np.array(suff_delta1).mean() if len(suff_delta1) != 0 else 1
                            logodd_i = np.array(logodd_delta1).mean() if len(logodd_delta1) != 0 else 0
                            # if args.cal_AOPC:
                            print("k:{} AOPC:{:.4f} Suff:{:.4f} LO:{:.4f} feature essence:{} feature minimality:{}".format(k,aopc_i,suff_i,logodd_i,flag_es1,flag_ms1))
                            # elif args.cal_suff:
                            # print("k:{} delta1:{} Suff:{}".format(k,suff_delta1,suff_i))
                            # elif args.cal_log:
                            # print("k:{} delta1:{} LO:{}".format(k,logodd_delta1,logodd_i))

                            delta2 = []
                            s_text2 = deepcopy(toks[i])
                            cnt_tok2 = cnt_tok
      
                            aopc_delta2,suff_delta2,logodd_delta2,flag_ms2 = calculate_metric(predictor,batch,i,label_id,probs,cnt_tok,ans2,aopc_token,pad_token,s_text,metric="AOPC",second=False)
                            
                            # suff_delta2 = sorted(suff_delta2)
                            # suff_delta2 = suff_delta2[:len(suff_delta2) * 0.2]

                            aopc_i2 = np.array(aopc_delta2).mean() if len(aopc_delta2) != 0 else 0
                            suff_i2 = np.array(suff_delta2).mean() if len(suff_delta2) != 0 else 1
                            logodd_i2 = np.array(logodd_delta2).mean() if len(logodd_delta2) != 0 else 0 
                            print("k:{} AOPC:{:.4f} Suff:{:.4f} LO:{:.4f} feature essence:{} feature minimality:{}".format(k,aopc_i2,suff_i2,logodd_i2,flag_es2,flag_ms2))
                           
                            # if args.cal_AOPC:
                            #     print("k:{} delta2:{} AOPC:{}".format(k,delta2,aopc_i2))
                            # elif args.cal_suff:
                            #     print("k:{} delta2:{} Suff:{}".format(k,delta2,aopc_i2))
                            # elif args.cal_log:
                            #     print("k:{} delta2:{} LO:{}".format(k,delta2,aopc_i2))                    
                        else:

                            # def calculate_CIG_norefine(pos_idx,grad_table,i):

                            if len(pos_idx) == 1:
                                ans = [((toks[i][pos_idx[0]],toks[i][pos_idx[0]]),1)] 
                                ans2 = []
                            else:
                                if args.no_refine:
                                    ans = calculate_CIG_norefine(pos_idx,grad_table,i,toks)
                                    ans2 = []
                                else:
                            # def calculate_DIG_second(knapsack,args,pos_idx,grad_table,upper_bound,second_order_ig_all,toks):
                                    ans,ans2 = calculate_DIG_second(knapsack,args,pos_idx,grad_table,upper_bound,second_order_ig_all,toks,i)
                            # print("dynamic feature set1:{}".format(ans))
                            # print("dynamic feature set2:{}".format(ans2))
                            

                            print("dynamic feature set1:{}".format(ans))
                            print("dynamic feature set2:{}".format(ans2))
                            aopc2 = []
                            cnt_tok = 0
                            if args.model_type == 'bert' or args.model_type == "distilbert":
                                cnt_tok = sum([1 if tok_ != "[PAD]" else 0 for tok_ in toks[i]])
                            else:
                                cnt_tok = sum([1 if tok_ != "<pad>" else 0 for tok_ in toks[i]])
                            s_text = deepcopy(toks[i])
                            print("cnt:{} raw token:{}".format(cnt_tok, s_text))
                            # def calculate_metric(predictor,batch,i,label_id,probs,cnt_tok,ans,aopc_token,pad_token,s_text,metric="AOPC"):

                            # if args.cal_log:
                            cnt_tok1 = cnt_tok
                            
                            aopc_delta1, suff_delta1, logodd_delta1 = [], [], []
                            flag_es1 = False
                            flag_ms1 = False
                            # flag1_1,flag1_2,flag2_1,flag2_2 = False,False,False,False
                            if len(ans) != 0:
                                aopc_delta1,suff_delta1,logodd_delta1,flag_ms1 = calculate_metric(predictor,batch,i,label_id,probs,cnt_tok,ans,aopc_token,pad_token,s_text,metric="AOPC")
                                

                                max_aopc = max(aopc_delta1)
                                if max_aopc >= probs[i,label_id] - 0.5:
                                    flag_es1 = True
                                # else:
                                    
    
                               

                            
                            aopc_i = np.array(aopc_delta1).mean() if len(aopc_delta1) != 0 else 0
                            suff_i = np.array(suff_delta1).mean() if len(suff_delta1) != 0 else 1
                            logodd_i = np.array(logodd_delta1).mean() if len(logodd_delta1) != 0 else 0
                            
                            print("k:{} AOPC:{:.4f} suff:{:.4f} LO:{:.4f}".format(k,aopc_i,suff_i,logodd_i))

                            aopc_delta2 = []
                            logodd_delta2 = []
                            suff_delta2 = []
                            flag_es2 = False
                            flag_ms2 = False
                            if len(ans2) != 0:
                                aopc_delta2,suff_delta2,logodd_delta2,flag_ms2 = calculate_metric(predictor,batch,i,label_id,probs,cnt_tok,ans2,aopc_token,pad_token,s_text,metric="AOPC")
                                # suff_delta2 = sorted(suff_delta2)
                                # suff_delta2 = suff_delta2[:int(len(suff_delta2) * 0.5)]
                                max_aopc = max(aopc_delta2)
                                if max_aopc >= probs[i,label_id] - 0.5:
                                    flag_es2 = True
                                # else:
                                    
                                # delta2 = delta2[:n_e]
                            aopc_i2 = np.array(aopc_delta2).mean() if len(aopc_delta2) != 0 else 0
                            suff_i2 = np.array(suff_delta2).mean() if len(suff_delta2) != 0 else 1
                            logodd_i2 = np.array(logodd_delta2).mean() if len(logodd_delta2) != 0 else 0
                            print("k:{} AOPC:{:.4f} suff:{:.4f} LO:{:.4f}".format(k,aopc_i2,suff_i2,logodd_i2))

                            


                    # if args.cal_AOPC:
                    #     res2.append(max(aopc_i,aopc_i2))
                    # elif args.cal_log or args.cal_suff:
                    #     res2.append(min(aopc_i,aopc_i2))
                aopc_res.append(max(aopc_i,aopc_i2))
                suff_res.append(min(suff_i,suff_i2))
                log_res.append(min(logodd_i,logodd_i2))

                
                    
            # if args.use_IG:

                if len(aopc_res) >= args.evaluate_amount:
                    break
            if len(aopc_res) >= args.evaluate_amount:
                    break 
                    
            # if args.useDIG:
                


    # print(len(res))
    # if args.use_IG:
    print(len(aopc_res))
    # print("feature_minimality_amount:{} feature_essence_amount:{}".format(feature_minimality_amount,feature_essence_amount))
    return np.array(aopc_res).mean() , np.array(suff_res).mean(),np.array(log_res).mean()
    # elif args.useDIG:
        # print(len(res2))
        # if cal_log:
        #     print("average LO:{:.4f}".format())
        # return np.array(res2).mean()
    
    
def evaluate(args, model, eval_dataset,example_book,tokenizer):
    # cal_mininum_feature(args,eval_dataset,model,example_book)
    if not args.only_eval:
        aopc,suff,lo = cal_AOPC(args,eval_dataset,model,example_book,tokenizer)
        # if args.cal_log:
        print("Final LO:{:.4f} Final AOPC:{:.4f} Final Suff:{:.4f}".format(lo,aopc,suff))
        # elif args.cal_AOPC:
            # print("Final AOPC:{}".format(aopc))
        # elif args.cal_suff:
            # print("Final SUFF:{}".format(aopc))

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
    remain_acc = 0
    # use_IG = args.use_IG
    # select_all = []
    # eps = 0.05
    if args.cal_acc:
        with open("delta_token.pkl","rb") as f:
            delta_token = pickle.load(f)
        delta_token1 = [d[0] for d in delta_token]
        delta_token2 = [d[1] for d in delta_token]
        vocab = tokenizer.get_vocab()
        # print(delta_token1,delta_token2)[(('ridiculous', '[PAD]'), 0.0)]

    delta_raw_token = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

                # if abs(word_ig[i][j].item() - )
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            if args.cal_acc:
                inputids = batch[4].detach().cpu().numpy().tolist()
                for j,input_id_i in enumerate(inputids):
                    if input_id_i in delta_token1:
                        # delta_raw_token.append()
                        input_token = batch[0][j].detach().cpu().numpy().tolist()
                        input_idx = delta_token1.index(input_id_i)
                        # [(tok1,tok2)]
                        remain_token = delta_token2[input_idx]
                        # [(('no', '[PAD]'), 0.0)]
                        # print(remain_token)
                        remain_tokena = [vocab[pi[0]] for pi,pj in remain_token]
                        remain_tokenb = [vocab[pi[1]] for pi,pj in remain_token]
                        # input_token = [tok_id if tok_id in remain_tokena and tok_id in remain_tokenb]
                        input_token_remain = []
                        mask = []
                        if args.model_type in ['bert', 'xlnet']:
                            token_type = []
                        # token
                        for tok_id in input_token:
                            if tok_id in remain_tokena and tok_id in remain_tokenb:
                                input_token_remain.append(tok_id)
                                mask.append(1)
                                if args.model_type in ['bert', 'xlnet']:
                                    token_type.append(0)
                        input_token_remain = [101] + input_token_remain + [102]
                        mask = [1] + mask + [1]
                        if args.model_type in ['bert', 'xlnet']:
                            token_type = [1] + token_type + [1]
                        input_token_remain = torch.LongTensor(input_token_remain).unsqueeze(0).cuda()
                        mask = torch.LongTensor(mask).unsqueeze(0).cuda()
                        if args.model_type in ['bert', 'xlnet']:
                            token_type = torch.LongTensor(token_type).unsqueeze(0).cuda()
                        outputs = model(input_ids = input_token_remain,
                                    token_type_ids = token_type if args.model_type in ['bert', 'xlnet'] else None ,
                                    attention_mask = mask,
                                    labels = batch[3][j])
                        _,logits = outputs[:2]
                        logits = torch.softmax(logits,dim=-1)
                        _ , pred = torch.max(logits,dim=1)
                        if pred[0].item() == batch[3][j].item():
                            remain_acc += 1



            if args.use_VMASK:
                outputs = model(inputs, 'train')
            else:
                outputs = model(input_ids = batch[0],
                                token_type_ids = batch[2] if args.model_type in ['bert', 'xlnet'] else None ,
                                attention_mask = batch[1],
                                labels = batch[3])
            
                    
            
            # outputs = model(inputs, 'eval')
            tmp_eval_loss, logits = outputs[:2]
            logits = torch.softmax(logits,dim=-1)
            
        
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
    if args.cal_acc:
        remain_acc = remain_acc / args.evaluate_amount
        print("Remain Acc:{:.3f}".format(remain_acc))
    
    return eval_loss, eval_acc


def load_and_cache_examples(args, task, tokenizer, type):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    reload = args.reload
    if os.path.exists(cached_features_file) and not reload:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        with open(os.path.join(args.data_dir,"example_book.pickle"),"rb") as f:
            example_book = pickle.load(f)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
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
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        with open(os.path.join(args.data_dir,"example_book.pickle"),"wb") as f:
            pickle.dump(example_book,f)
        logger.info("Saving example book into cached file %s", os.path.join(args.data_dir,"example_book.pickle"))


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_ids = torch.tensor([f.id for f in features],dtype=torch.int)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_ids)
    return dataset,example_book


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='./dataset/data_name', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
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
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
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
    parser.add_argument("--useDIG",action="store_true")
    parser.add_argument("--use_VMASK",action="store_true")
    parser.add_argument("--cal_AOPC",action="store_true")
    parser.add_argument("--use_lime",action="store_true")
    parser.add_argument("--k",type=int,default=10)
    parser.add_argument("--iter_epochs",type=int,default=10)
    parser.add_argument("--step_lr",type=float,default=0.1)
    parser.add_argument("--threshold",type=float,default=465)
    parser.add_argument("--alpha",type=float,default = 465)
    parser.add_argument("--evaluate_amount" , type = int, default = 500)
    parser.add_argument("--pad_token",type=str,default="pad")
    parser.add_argument("--static",type=int,default = 0)
    parser.add_argument("--step_lr1",type=float,default=0.1)
    parser.add_argument("--step_lr2",type=float,default=0.5)
    parser.add_argument("--value_type",type = str, default = "linear")
    parser.add_argument("--encoder_type",type=str,default="bert")
    parser.add_argument("--mu",type=float,default=0.1)
    parser.add_argument("--use_regular", action="store_true")
    parser.add_argument("--lambda_co",type=float,default=0.1)
    parser.add_argument("--use_cls",action="store_true")
    parser.add_argument("--is_second",action="store_true")
    parser.add_argument("--cal_acc",action="store_true")
    parser.add_argument("--only_eval",action="store_true")
    parser.add_argument("--cal_log",action="store_true")
    parser.add_argument("--cal_suff",action="store_true")
    parser.add_argument("--no_refine",action="store_true")

    # parser.add_argument("--maxL",type=int,default=)
    

    args = parser.parse_args()
    
    # if not os.path.exists("")
    log_path = os.path.join(args.log_dir,args.task_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(log_path,"output.log"))],
    )
    logger.setLevel(logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

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
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # model_class, tokenizer_class = None, None
    logger.info("Training/evaluation parameters %s", args)

    # Load pretrained bert model
    bert_path = args.bert_path
    tokenizer_path = args.tokenizer_path
    # prebert = bertEncoder(args)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)
    
    
    if args.model_name_or_path == "bert-base-uncased":
        model = bertEncoder(args)
    elif args.model_name_or_path == "roberta-base":
        # print(args.model_name_or_path)
        model = robertaEncoder(args)

    elif args.model_name_or_path == "distilbert-base-uncased":
        model = distilbertEncoder(args)
    model.to(args.device)
    # print(model)
    # fix embeddings
    # parameters = filter(lambda p: p.requires_grad, model.bertmodel.bert.embeddings.parameters())
    # for param in parameters:
    #     param.requires_grad = False


    
    with open(os.path.join(args.output_dir, args.savename), 'rb') as f:
        print(args.output_dir,args.savename)
        model = torch.load(f)
        print(model)
    model.to(torch.device(args.device))
    # print(model)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    # Test
    test_dataset,example_book = load_and_cache_examples(args, args.task_name, tokenizer, type='test')
    # if args.cal_AOPC:
    #     # print(model)
    #     test_loss, test_acc,aopc = evaluate(args, model, test_dataset,example_book,tokenizer)
    #     logger.info('\ntest_loss {:.6f} | test_acc {:.6f} aopc:{:.6f}'.format(test_loss, test_acc,aopc))
    # else:
    test_loss,test_acc = evaluate(args,model,test_dataset,example_book,tokenizer)
    logger.info('\ntest_loss {:.6f} | test_acc {:.6f}'.format(test_loss, test_acc))

    # cal_IG(args,test_dataset,model,example_book)

    # return test_loss, test_acc


if __name__ == "__main__":
    main()
