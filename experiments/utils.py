import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.distributions import Categorical


def read_vocab(filepath):
    vocab = []
    vector_list = []
    with open(filepath,"r") as f:
        data = f.readlines()
    # data = data[1:]
    for line in data:
        line = line.strip()
        line = line.split()
        word = line[0]
        vector = line[1:]
        vocab.append(word)
        vector_list.append(np.array(vector,dtype=np.float32))
        # print(vector_list[-1].shape)
        # if vector_list[-1].shape[0] != 100:
        #     print("bad")
        #     print(word)
    vocab = ['pad','unk'] + vocab
    unk_vector = np.random.normal(0,1,300)
    pad_vector = np.zeros(300,dtype=np.float32)
    
    vector_list = [pad_vector] + [unk_vector] + vector_list
    # print(vector_list[0].shape,vector_list[-1].shape)
    vector_list = np.stack(vector_list)

    return vocab,vector_list



def select_action(policy,grad_state, sent_state = None):
    # sent_state : B,L,D
    # grad_state : B,L
    # state = torch.from_numpy(state).float()
    probs = policy(sent_state,grad_state) # B,L,2

    m = Categorical(probs)

    action = m.sample()

    policy.saved_log_probs.append(m.log_prob(action))

    return action  # B

def finish_episode(policy,args,optimizer,eps = 1e-3):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0,R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs,returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
# def reward(probs,)
