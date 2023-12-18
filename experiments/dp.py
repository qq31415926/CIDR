from captum.attr import ShapleyValueSampling
import numpy as np
import torch
from itertools import cycle,permutations,chain,combinations
from collections import OrderedDict
from functools import total_ordering
import math
import torch.nn as nn
from copy import deepcopy


def cal_shapley_value(features,preds,model):
    '''
    features:B,N
    '''
    explainer = ShapleyValueSampling(model)
    attr = explainer.attribute(features,target=preds)
    return attr
def compute_shapley_score(partition_index, rand_rows, row_to_investigate, model, weight_col_name=None):
    # type: (int, Iterable[Row], Row, ShparkleyModel, Optional[str]) -> Generator[Tuple[str, float, float], None, None]
    """
    Computes the shapley marginal contribution for each feature in the feature vector over all
    samples in the partition.
    The algorithm is based on a monte-carlo approximation:
    https://christophm.github.io/interpretable-ml-book/shapley.html#fn42
    :param partition_index: Index of spark partition which will serve as a seed to numpy
    :param rand_rows: Sampled rows of the dataset in the partition
    :param row_to_investigate: Feature vector for which we need to compute shapley scores
    :param model: ShparkleyModel object which implements the predict function.
    :param weight_col_name: column name with row weights to use when sampling the training set
    :return: Generator of tuple of feature and shapley marginal contribution
    """
    required_features = list(model.get_required_features())
    random_feature_permutation = np.random.RandomState(partition_index).permutation(required_features)
    # We cycle through permutations in cases where the number of samples is more than
    # the number of features
    permutation_iter = cycle(permutations(random_feature_permutation))
    feature_vector_rows = []
    rand_row_weights = []
    for rand_row in rand_rows:  # take sample z from training set
        rand_row_weights.append(rand_row[weight_col_name] if weight_col_name is not None else 1)
        feature_permutation = next(permutation_iter)  # choose permutation o
        # gather: {z_1, ..., z_p}
        feat_vec_without_feature = OrderedDict([(feat_name, rand_row[feat_name]) for feat_name in required_features])
        feat_vec_with_feature = feat_vec_without_feature.copy()
        for feat_name in feature_permutation:  # for random feature k.
            # x_+k = {x_1, ..., x_k, .. z_p}
            feat_vec_with_feature[feat_name] = row_to_investigate[feat_name]
            # x_-k = {x_1, ..., x_{k-1}, z_k, ..., z_p}
            # store (x_+k, x_-k)
            feature_vector_rows.append(feat_vec_with_feature.copy())
            feature_vector_rows.append(feat_vec_without_feature.copy())
            # (x_-k = x_+k)
            feat_vec_without_feature[feat_name] = row_to_investigate[feat_name]

    if len(feature_vector_rows) == 0:
        return
    preds = model.predict(feature_vector_rows)
    feature_iterator = chain.from_iterable(cycle(permutations(random_feature_permutation)))
    for pred_index, feature in zip((range(0, len(preds), 2)), feature_iterator):
        marginal_contribution = preds[pred_index] - preds[pred_index + 1]

        #  There is one weight added per random row visited.
        #  For each random row visit, we generate 2 predictions for each required feature.
        #  Therefore, to get index into rand_row_weights, we need to divide
        #  prediction index by 2 * number of features, and take the floor of this.
        weight = rand_row_weights[pred_index // (len(required_features) * 2)]

        yield (str(feature), float(marginal_contribution), float(weight))





    
def minimum_feature_set(w,v,n,c):
    '''
    weights: N
    values:N
    n:number of positive features
    c:
    '''
    # print(w,v,n,c)

    # print(v)
    # print(w)
    # print(v)
    w = [round(wi,3) for wi in w]
    v = [round(vi,3) for vi in v]
    print("w:",w)
    print("v:",v)
    print("n:",n)
    print("c:",c)
    
    
    c = round(c,3)
    p = 10000 * [[0.0,0.0]]
    p[0][0] = 0.0
    p[0][1] = 0.0
    left = 0 
    right = 0 
    next = 1
    # print(n)
    head = (n + 2) * [0]
    head[n + 1] = 0
    head[n] = 1
    for i in range(n - 1,-1,-1):
        k = left
        for j in range(left,right + 1):
            if p[j][0] + w[i] > c:
                break
            nw = p[j][0] + w[i]
            nv = p[j][1] + v[i]

            while k <= right and p[k][0] < nw :
                p[next][0] = p[k][0]
                p[next][1] = p[k][1]
                k += 1
                next += 1
            if k <= right and p[k][0] == nw:
                if p[k][1] > nv:
                    nv = p[k][1]
                k += 1
            if nv > p[next - 1][1]:
                p[next][0] = nw
                p[next][1] = nv
                next += 1
            while k <= right and p[k][1] < nv :
                k += 1

        while k <= right:
            p[next][0] = p[k][0]
            p[next][1] = p[k][1]
            k += 1
            next += 1
        left = right + 1
        right = next - 1
        head[i] = next
    trace = traceBack(v,w,p,head)
    # print("minimum feature set:{}".format(traceBack(v,w,p,head)))
    return trace
    
            
   
def traceBack(v,w,p,head):
    trace = []
    k = head[0] - 1
    n = len(w)
    for i in range(1,n + 1):
        left = head[i + 1]
        right = head[i] - 1
        for j in range(left,right + 1):
            if p[j][0] + w[i-1] == p[k][0] and p[j][1] + v[i - 1] == p[k][1]:
                k = j
                trace.append(i)
                break
    return trace

    
  
        

def integrated_gradients(model, text_token, token_mask, y,model_type = "bert",is_second = False,attr_token = None,attr_mask = None):
    '''
    用于计算NLP模型中的积分梯度；
    1、由于NLP是离散型输入，因此只能通过对embedding layer的权重进行线性插值来实现输入的线性插值
    2、计算之后得到的结果是（input_len,dim），计算每一个词向量累加和当做词的重要性
    :return:
    '''
    # print(y)
    # 除embedding层外，固定住所有的模型参数
    for name, weight in model.named_parameters():
        if 'embedding' not in name:
            weight.requires_grad = False

    # 获取原始的embedding权重
    # init_embed_weight = model.word_attn.embedding.weight.data
    if model_type == "bert":
        init_embed_weight = model.bertmodel.embeddings.word_embeddings.weight.data
    elif model_type == "roberta":
        init_embed_weight = model.roberta.embeddings.word_embeddings.weight.data
    elif model_type == "distilbert":
        init_embed_weight = model.distilbert_model.embeddings.word_embeddings.weight.data

    x = text_token # B,N

    # 获取输入之后的embedding
    init_word_embedding = init_embed_weight[x] # B,N,D
    # print(init_word_embedding.size())

    # 获取baseline
    baseline = 0 * init_embed_weight
    baseline_word_embedding = baseline[x] # B,N,D

    # 计算线性路径积分
    steps = 20
    # 对目标权重进行线性缩放计算的路径
    gradient_list = []

    for i in range(steps + 1):
        # 进行线性缩放
        scale_weight = baseline + float(i / steps) * (init_embed_weight - baseline)

        # 更换模型embedding的权重
        if model_type == "bert":
            model.bertmodel.embeddings.word_embeddings.weight.data = scale_weight
        elif model_type == "roberta":
            model.roberta.embeddings.word_embeddings.weight.data = scale_weight
        elif model_type == "distilbert":
            model.distilbert_model.embeddings.word_embeddings.weight.data = scale_weight

            

        # 前馈计算
        pred = model(input_ids=x, attention_mask=token_mask) # B,C
        if type(pred) == tuple:
            pred = pred[0]
        
        # print(pred)
        # 直接取对应维度的输出(没经过softmax) 
        target_pred = pred[:, y] # B
        # print(target_pred)

        # 计算梯度
        if not is_second:
            target_pred.sum().backward() 
        else:
            target_pred.sum().backward()
        # print(model.bertmodel.embeddings.word_embeddings.weight.grad_fn)
        # 获取输入变量的梯度
        if model_type == "bert":
            grads = model.bertmodel.embeddings.word_embeddings.weight.grad[x].unsqueeze(1)
        elif model_type == "roberta":
            grads = model.roberta.embeddings.word_embeddings.weight.grad[x].unsqueeze(1)
        elif model_type == "distilbert":
            grads = model.distilbert_model.embeddings.word_embeddings.weight.grad[x].unsqueeze(1)
        # print(grads.shape)
        # grads = grads
        gradient_list.append(grads)
        # print(gradient_list[-1])
        # 梯度清零，防止累加
        model.zero_grad()

    # steps,input_len,dim
    gradient_list = torch.cat(gradient_list,dim=1)# B,S,L,D
    # gradient_list = torch.from_numpy(np.asarray(gradient_list)) 
    
    # input_len,dim
    avg_gradient = torch.mean(gradient_list,dim=1)# B,L,D
    # avg_gradient = avg_gradient.detach().cpu().numpy()
    # x-baseline
    delta_x = init_word_embedding - baseline_word_embedding
    # delta_x = delta_x.detach().cpu().numpy()
    # print(delta_x.shape)

    # 获取积分梯度
    ig = avg_gradient * delta_x

    # 对每一行进行相加得到(input_len,)
    # word_ig = np.sum(ig, axis=1)
    word_ig = torch.sum(ig,dim=-1) # B,L
    model.zero_grad()
    if is_second:

        L = x.shape[1]
        # attr_token B,L,L-1 

        attr_token = torch.LongTensor(attr_token).cuda()
        # print(attr_mask)
        attr_mask = torch.IntTensor(attr_mask).cuda()
        # x = x.unsqueeze(-1).expand(-1,-1,L - 1) # B,L,L-1
    # print(init_word_embedding.size())

    # 获取baseline
        baseline = 0 * init_embed_weight # B,V,D
        grad_table = [] # L,S,B,L-1
        for ll in range(L):
            attr_token_ = attr_token[:,ll,:] # B,L-1
            attr_mask_ = attr_mask[:,ll,:] # B,L-1
            init_word_embedding = init_embed_weight[attr_token_] # B,L-1,D
            baseline_word_embedding = baseline[attr_token_] # B,L-1,D

            grad_table_l = []
            for i in range(steps + 1):
                # B,V,D
                scale_weight = baseline + float(i / steps) * (init_embed_weight - baseline)

               
                if model_type == "bert":
                    model.bertmodel.embeddings.word_embeddings.weight.data = scale_weight
                elif model_type == "roberta":
                    model.roberta.embeddings.word_embeddings.weight.data = scale_weight
                elif model_type == "distilbert":
                    model.distilbert_model.embeddings.word_embeddings.weight.data = scale_weight

                
                pred = model(input_ids=attr_token_, attention_mask=attr_mask_) # B,C
                if type(pred) == tuple:
                    pred = pred[0]
                
                # print(pred)
                # 直接取对应维度的输出(没经过softmax) 
                target_pred = pred[:, y] # B
                # print(target_pred)

                # 计算梯度
                target_pred.sum().backward() 
                
                if model_type == "bert":

                    grads = model.bertmodel.embeddings.word_embeddings.weight.grad[attr_token_].unsqueeze(1)
                elif model_type == "roberta":
                    grads = model.roberta.embeddings.word_embeddings.weight.grad[attr_token_].unsqueeze(1)
                elif model_type == "distilbert":
                    grads = model.distilbert_model.embeddings.word_embeddings.weight.grad[attr_token_].unsqueeze(1)
                # print("grad shape:{}".format(grads.shape))
                
                # B,1,L-1,D
                grad_table_l.append(grads) 
                # print(gradient_list[-1])
                # 梯度清零，防止累加
                model.zero_grad()
            
            grad_table_l = torch.cat(grad_table_l,dim=1) # B,S,L-1,D
            avg_gradient = torch.mean(grad_table_l,dim=1) # B,L-1,D
            # print("avg gradient shape {}".format(avg_gradient.shape))
            delta_x = init_word_embedding - baseline_word_embedding # B,L-1,D
            # print("delta x shape {}".format(delta_x.shape))
            ig = avg_gradient * delta_x # B,L-1,D
            word_ig_one_token = torch.mean(ig,dim=-1) # B,L-1
            # print(word_ig_one_token.shape)
            del attr_token_
            del attr_mask_
            
            grad_table.append(word_ig_one_token.detach().cpu().numpy()) # L,B,L-1
        # print("grad_table:{}".format(grad_table))
        grad_table = torch.tensor(grad_table)
        # print(grad_table.shape) # L,B,L-1
        # grad_table = torch.sum(grad_table,dim=-1)# L,B,L-1
        grad_table = grad_table.permute(1,0,2) # B,L,L-1

        # grad_table = grad_table.permute(2,0,3,1) #  L,S,B,L-1 -> B,L,L-1,S
        # grad_table = torch.mean(grad_table,dim = -1) # B,L,L-1


    model.zero_grad()
    if is_second:
        return word_ig,grad_table
    else:
        return word_ig


def multi_order_integrated_gradients():
    pass
def print_word_ig(word_ig,input_ids,id2word,example_book,raw_id):
    '''
    word_ig:B,N
    '''
    B = word_ig.shape[0]
    L = word_ig.shape[1]
    
    for i in range(B):
        print("sentence:{} length:{}".format(example_book[raw_id[i].item()],L))
        # print()
        word_igi = word_ig[i,:]
        for j in range(L):
            print("token:{} Integrated Gradients:{:.5f}".format(id2word[input_ids[i,j].item()],word_igi[j].item()))
        break


def knapsack(w, v, n, c, p = 2,maxL=301):
    w = [ round(wi , p) for wi in w ]
    # v = [ round(vi , p) for vi in v ]
    c = round(c, p)
    l = len(str(c).split('.')[0])
    c = int(c * math.pow(10, p))
    w = [int(wi * math.pow(10, p)) for wi in w]
    
    
    f = [0.0] * int(math.pow(10 , l + p))
    path = maxL * [[0] * int(math.pow(10 , l + p))] 
    for i in range(n):
        for j in range(c , w[i] - 1,-1):
            if f[j] < f[j - w[i]] + v[i]:
                f[j] = f[j - w[i]] + v[i]
                path[i][j] = 1
    def trace_path():
        i = n - 1
        j = c
        res = []
        # print(type(i),type(j))
        while i >= 0 and j >= 0:
            if path[i][j] == 1:

                res.append(i)
                j -= w[i]
            i -= 1
        return res
    res = trace_path()
    return res      

# def clip_grad_norm(grad)


def calculate_IG(predictor,args,word_ig,batch,toks,aopc_token,pad_token,probs,i,label_id,k):

    # ig_i = word_ig[i]
    ig_i = word_ig[i,:]
    ig_i = ig_i.detach().cpu().numpy().tolist()
    if args.model_type == "roberta":
        ig_i = ig_i[:-2]
    else:
        ig_i = ig_i[:-1]
    
            
    cnt_tok = 0
    # print("toks:{}".format(toks[i]))
    if args.model_type == 'bert' or args.model_type == "distilbert":
        cnt_tok = sum([1 if tok_ != "[PAD]" else 0 for tok_ in toks[i]])
    else:
        cnt_tok = sum([1 if tok_ != "<pad>" else 0 for tok_ in toks[i]])
    # print(len(toks[i])) 
    # print(len(ig_i)) 
    # if args.model_type == "roberta":
    #     word_score = {toks[i][j]:\
    #         ig_i[j]  for j in range(len(ig_i) - 1)}
    # else:
    word_score = {toks[i][j]:\
        ig_i[j]  for j in range(len(ig_i) )}
                    
    ans = sorted(word_score.items(), key = lambda x : x[1] , reverse = True)
    # print(ans)
    aopc = []
    ans = [z[0] for z in ans][:k]
    ans = ans[:k]

    print("len:{} answer set:{} cnt_tok:{}".format(len(ans),ans,cnt_tok))
    s_text = deepcopy(toks[i])
    s_text_s = deepcopy(toks[i])
    raw_text = deepcopy(toks[i])

    print("raw token:{} count:{}".format(s_text,cnt_tok))
    delta = []
    suff_set = [ans[j] for j in range(len(ans))]
    suff_set = list(set(suff_set))
    for tok_ in s_text_s:
        if tok_ not in suff_set:
            s_text_s.remove(tok_)
    mask_s = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_s]


    aopc_delta = []
    log_delta = []
    suff_delta = []
    flag_ms = True
    flag_es = False
    for j in range(len(ans)):
        s_text_ms = deepcopy(raw_text)
        cnt_ms = cnt_tok
        tok = ans[j]
        if aopc_token == "del":
            s_text.remove(tok)
            if tok not in ["[PAD]","<pad>"]:
                cnt_tok -= 1
            mask = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text]
            for tok_ in suff_set:
                if tok_ != tok:
                    s_text_ms.remove(tok_)
            if tok not in ["[PAD]","<pad>"]:
                cnt_ms -= 1
            mask_ms = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_ms]
                        
        else:
            for k in range(len(s_text)):
                if s_text[k] == tok:
                    s_text[k] = pad_token
                    # break
                    cnt_tok -= 1
                                    
            mask = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text ]
        with torch.no_grad():
            probs_is = predictor(s_text_s,mask_s,is_tokenize=False)
            if probs_is[0,label_id].item() <= 0.5:
                flag_es = True
            suff_i = probs[i,label_id] - probs_is[0,label_id]
            if cnt_tok != 0:
                probs_i = predictor(s_text,mask,is_tokenize=False) 
                aopc_i = probs[i,label_id]-probs_i[0,label_id]      
                log_i = -math.log(probs[i,label_id] / probs_i[0,label_id])      
            else:
                aopc_i = probs[i,label_id].item()
                log_i = -math.log(probs[i,label_id] / 0.01)

            probforms = 0
            # if cnt_ms != 0:
            probs_ms = predictor(s_text_ms,mask_ms,is_tokenize = False)
            probforms = probs_ms[0,label_id]
            if probforms < 0.5:
                flag_ms = False
                
                

        
        print("Remove:{} AOPC:{:.4f} SUFF:{:.4f} LO:{:.4f}".format(tok,aopc_i,suff_i,log_i))
        
        # delta.append(delta_i)
        aopc_delta.append(aopc_i)
        suff_delta.append(suff_i)
        log_delta.append(log_i)
                    
                            
    aopc_i = np.array(aopc_delta).mean()
    suff_i = np.array(suff_delta).mean()
    logodd_i = np.array(log_delta).mean()
    max_aopc = max(aopc_delta)
    # flag_es = True if max_aopc >

    return aopc_i,suff_i,logodd_i,flag_ms,flag_es
def calculate_CIG_norefine(pos_idx,grad_table,i,toks):
    two_pairs = combinations(pos_idx,2)
    dict_pair = {two_pair_idx : two_pair for two_pair_idx,two_pair in enumerate(two_pairs)}
    print("len dict pair{}".format(len(dict_pair)))
    wi = [grad_table[i][pi][pj].item() for pi,pj in combinations(pos_idx,2)]
    print("wi:{}".format(wi))
    # print("dict pair:{}".format(dict_pair))
    ans = [((toks[i][dict_pair[pair_][0]],toks[i][dict_pair[pair_][1]]),wi[j]) for j,pair_ in enumerate(dict_pair)]

    ans = sorted(ans, key = lambda x : x[1], reverse = True)
    
    return ans


def calculate_DIG_first(args,pos_idx, word_ig_i, word_pos, lower_bound, i, upper_bound,toks,):
    pos2raw = { ii : j for ii , j  in  enumerate(pos_idx) }
    # print("pos2raw:{}".format(pos2raw))
    sigmoid = nn.Sigmoid()

    # index_1 = torch.IntTensor(pos_idx).cuda()
    # word_pos = torch.index_select(word_ig_i, dim = 0 ,index = index_1) # postive feature
    # del index_1
    min_ig = torch.min(word_pos)
    l_bound = lower_bound[i] + min_ig
    wi = word_pos.detach().cpu().numpy().tolist()
    
    print("positive number:{}".format(len(wi)))
    
    vi = torch.normal(mean = 0, std = 1, size = (len(wi),)).cuda() # postive feature weight 
    
    
    
    result = []
    print("Example:{} lower bound:{} upper bound:{}".format(i,l_bound.item(),upper_bound[i].item()))
    upper_bound_i = upper_bound[i].item()
    for it in range(args.iter_epochs):
        # feature set:index of positive feature index from 1 to postive feature number
        # print(vi)
        print("Epoch:{}".format(it + 1))
        # vi = (vi - torch.min(vi) + 1e-5)/(torch.max(vi) - torch.min(vi))
        # if it == 0:
        vi = sigmoid(vi)
        # vi = softmax(vi)
        v = vi.detach().cpu().numpy().tolist()
        wi = word_pos.detach().cpu().numpy().tolist()
        print("v:{}".format(v))
        print("w:{}".format(wi))
        
        feature_set =  knapsack(        w = wi,
                                        v = v,
                                        n = len(wi),
                                        c = upper_bound_i)
        print("feature set:{}".format(feature_set))
        sum_of_feature = torch.sum(word_pos[feature_set]).item()
        print("sum of feature set:{} lower_bound:{}".format(sum_of_feature,l_bound.item()))
        ratio = len(feature_set) * 1.0 / len(v) 
        
        print("ratio:{:.3f} upper_bound:{:.3f}".format(ratio,upper_bound_i))
        beta = ratio * ratio - 2.5 * ratio + 2
        upper_bound_i *= beta
        
        
        
        
        if it == args.iter_epochs - 1:
            result = feature_set
    
    vi = vi.detach().cpu().numpy().tolist()
    other_result = [j for j in range(len(vi)) if j not in result]
    

    result = [pos2raw[j] for j in result]
    result_ig = [word_ig_i[j].item() for j in result]
    result = [toks[i][t] for t in result]
    
    other_result = [pos2raw[j] for j in other_result]
    other_result_ig = [word_ig_i[j].item() for j in other_result]
    # result = [toks[i][t] for t in result]
    other_result = [toks[i][t] for t in other_result]
    # print("result1:{} result2:{}".format(result,other_result))
    tok_score = {
        t : t_ig 
        for t,t_ig in zip(other_result, other_result_ig)
    }
    tok_score2 = {
        t : t_ig 
        for t, t_ig in zip(result, result_ig)
    }
    
    ans = sorted(tok_score.items() , key = lambda x: x[1] , reverse = True)
    ans2 = sorted(tok_score2.items() , key = lambda x: x[1] , reverse = True)

    return ans, ans2

def calculate_DIG_second(knapsack,args,pos_idx,grad_table,upper_bound,second_order_ig_all,toks,i):
    two_pairs = combinations(pos_idx,2)
    sigmoid = nn.Sigmoid()
               
    dict_pair = {two_pair_idx : two_pair for two_pair_idx,two_pair in enumerate(two_pairs)}
    print("len dict pair{}".format(len(dict_pair)))
    # positive_num = len(two_pairs)
    vi = torch.normal(mean = 0, std = 1, size = (len(dict_pair),)) # postive feature weight 
    # for pi,pj in two_pairs:
    #     print("second order ig {}".format(grad_table[i][pi][pj].item()))
    wi = [grad_table[i][pi][pj].item() for pi,pj in combinations(pos_idx,2)]
    print("wi:{}".format(wi))
    # w = torch.FloatTensor(wi).cuda()
    print("second order sum:{} second order upper bound:{}".format(sum(wi), upper_bound[i]))
    upper_bound_i = upper_bound[i]
    wi_min = np.min(np.array(wi))
    wi = [wi_ - wi_min for wi_ in wi]
    for it in range(args.iter_epochs):
        print("Epoch:{}".format(it + 1))
        # vi = (vi - torch.min(vi) + 1e-5)/(torch.max(vi) - torch.min(vi))
        # if it == 0:
        vi = sigmoid(vi)
        # vi = softmax(vi)
        v = vi.numpy().tolist()
        
        # wi = word_pos.detach().cpu().numpy().tolist()
        print("v:{}".format(v))
        print("w:{}".format(wi))
        # wi = [pw if pw > 0 else 1e-4 for pw in wi]
        feature_set =  knapsack(        w = wi,
                                        v = v,
                                        n = len(wi),
                                        c = upper_bound_i,
                                        maxL = len(wi) + 1)
        print("feature set:{}".format(feature_set))
        # sum_of_feature = torch.sum(word_pos[feature_set]).item()
        # print("sum of feature set:{} lower_bound:{}".format(sum_of_feature,l_bound.item()))
        ratio = len(feature_set) * 1.0 / len(v) 
        
        print("ratio:{:.3f} upper_bound:{:.3f}".format(ratio,upper_bound_i))
        beta = ratio * ratio - 2.5 * ratio + 2
        # beta = ratio * ratio * 0.1 - 2.05 * ratio + 2
        # beta /= 10
        upper_bound_i *= beta

        
        if it == args.iter_epochs - 1:
            result = feature_set
    
    vi = vi.cpu().numpy().tolist()
    other_result = [j for j in range(len(vi)) if j not in result]

    result = [dict_pair[j] for j in result]
    result_ig = [second_order_ig_all[i][pi][pj] for pi,pj in result]
    result = [(toks[i][pi],toks[i][pj]) for pi,pj in result]

    other_result = [dict_pair[j] for j in other_result]
    other_result_ig = [second_order_ig_all[i][pi][pj] for pi,pj in other_result]
    other_result = [(toks[i][pi],toks[i][pj]) for pi,pj in other_result]
    tok_score = {
        t : t_ig 
        for t,t_ig in zip(other_result, other_result_ig)
    }
    tok_score2 = {
        t : t_ig 
        for t, t_ig in zip(result, result_ig)
    }
    ans = sorted(tok_score.items() , key = lambda x: x[1] , reverse = True)
    ans2 = sorted(tok_score2.items() , key = lambda x: x[1] , reverse = True)
    return ans,ans2
    
def calculate_metric(predictor,batch,i,label_id,probs,cnt_tok,ans,aopc_token,pad_token,s_text,metric="AOPC",second=True):
    delta1 = []
    print("ground truth:{} pred:{} prob:{}".format(batch[3][i].item(),label_id,probs[i,label_id].item()))
    cnt_tok1 = cnt_tok
    aopc_delta1 = []
    suff_delta1 = []
    logodd_delta1 = []
    if second:
        suff_set = [ans[j][0][0] for j in range(len(ans))] + [ans[j][0][1] for j in range(len(ans))]
        suff_set = list(set(suff_set))
    else:
        suff_set = [ans[j][0] for j in range(len(ans))]
    s_text_s = deepcopy(s_text)
    raw_text = deepcopy(s_text)

    if aopc_token == "del":
        for tok_ in s_text_s:
            if (not tok_ in suff_set) and tok_ != "[PAD]":
                s_text_s.remove(tok_)
        mask_s = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_s]
    else:
        for s_text_tok in range(len(s_text_s)):
            if not (s_text_s[s_text_tok] in suff_set ):
                s_text_s[s_text_tok] = pad_token
        mask_s = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text ]

    # suff_set = []
    if metric == "AOPC":
        flag_ms = True
        flag_es = False
        # tok_ms = []
        for j in range(len(ans)):
            # print(ans[j])
            s_text_ms = deepcopy(raw_text)
            # print("s_text_ms:{}".format(s_text_ms))

            cnt_ms = cnt_tok1
            if second:
                tok1 = ans[j][0][0]
                tok2 = ans[j][0][1]
                

                if aopc_token == "del":
                    if tok1  in s_text:
                        s_text.remove(tok1)
                        if not (tok1 == '[PAD]' or tok1 == "<pad>"):
                            cnt_tok1 -= 1
                    if tok2  in s_text:
                        s_text.remove(tok2)
                        if not (tok2 == '[PAD]' or tok2 == "<pad>"):
                            cnt_tok1 -= 1
                    for tok_ in suff_set:
                        
                        if tok_ != tok1 and tok_ != tok2:
                            # print("s_text_ms:{} tok_:{}".format(s_text_ms,tok_))
                            # print(tok_) # 
                            s_text_ms.remove(tok_)
                    # if tok1 in s_text_ms:
                    #     s_text_ms.remove(tok1)
                    #     if not (tok1 == '[PAD]' or tok1 == "<pad>"):
                    #         cnt_ms -= 1
                    # if tok2 in s_text_ms:
                    #     s_text_ms.remove(tok2)
                    #     if not (tok2 == '[PAD]' or tok2 == "<pad>"):
                    #         cnt_ms -= 1
                        # s_text_ms.remove(tok2)
                        

                    mask = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text]
                    mask_ms = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_ms]
                else:
                    # s_text = [pad_token  if ll == tok else ll for ll in s_text]
                    for s_text_tok in range(len(s_text)):
                        
                        if s_text[s_text_tok] == tok1 :
                            s_text[s_text_tok] = pad_token
                            # if 
                            cnt_tok1 -= 1
                        elif s_text[s_text_tok] == tok2:
                            s_text[s_text_tok] = pad_token
                            cnt_tok1 -= 1
                            
                    mask = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text ]
            else:
                tok = ans[j][0]
             
                if aopc_token == "del":
                    s_text.remove(tok)
                    s_text_ms.remove(tok)
                    if not (tok == "[PAD]" or tok == "<pad>"):
                        cnt_tok1 -= 1
                    mask = [0 if ll == pad_token  or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text]
                    mask_ms = [0 if ll == pad_token  or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_ms]

                else:
                    for s_text_tok in range(len(s_text)):
                        if s_text[s_text_tok] == tok:
                            s_text[s_text_tok] = pad_token
                            # break
                            cnt_tok1 -= 1
                            
                    mask = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text ]
                
            
            with torch.no_grad():
                probs_is = predictor(s_text_s,mask_s,is_tokenize = False)
                if flag_es != True and probs_is[0,label_id].item() >= 0.5:
                    flag_es = True
                suff_delta1.append(probs[i,label_id] - probs_is[0,label_id])
                if cnt_tok1 == 0:
                    # delta1.append(probs[i,label_id]) 
                    aopc_delta1.append(probs[i,label_id])
                    logodd_delta1.append(-math.log(probs[i,label_id] / 0.001))
                    # print("token {} aopc")
                    break
                else:
                    probs_i = predictor(s_text,mask,is_tokenize = False) 
                    # delta1.append(probs[i,label_id]-probs_i[0,label_id]) 
                    aopc_delta1.append(probs[i,label_id]-probs_i[0,label_id])
                    logodd_delta1.append(-math.log(probs[i,label_id]/probs_i[0,label_id]))
                # if cnt_ms == 0:
                #     flag_ms = True
                probforms = 0
                if cnt_ms != 0:
                    probs_ms = predictor(s_text_ms,mask_ms,is_tokenize = False)
                    probforms = probs_ms[0,label_id]
                    if probforms < 0.5:
                        flag_ms = False
                else:
                    flag_ms = True
                
            if second:
                print("token_pair {}--{}  aopc:{:.4f} suff:{:.4f} LO:{:.4f} MS_Prob:{:.4f} tokens:{} ".format(tok1, tok2, aopc_delta1[-1], suff_delta1[-1],  logodd_delta1[-1], probforms, s_text))
            else:
                print("token {}  aopc:{:.4f} suff:{:.4f} LO:{:.4f} MS_Prob:{:.4f} tokens:{} ".format(tok, aopc_delta1[-1],suff_delta1[-1], logodd_delta1[-1], probforms, s_text))

    
    return aopc_delta1,suff_delta1,logodd_delta1,flag_ms,flag_es