import os
import random
from collections import defaultdict

def split_dataset(datadir, dataset, mode = "train",des = None):
    dataname = os.path.join(datadir,dataset)
    
    with open(os.path.join(dataname,  "{}.tsv".format(mode)),"r") as f:
        data = f.readlines()
    data = data[1:]
    sentences = []
    labels = []
    for l in data:
        t = l.strip().split()
        sentences.append(" ".join(t[:-1]))

        labels.append(t[-1])

    all_labels = list(set(labels))

    small_label = ["0","1"]
    sample_sentences = []
    sample_labels = []
    for sentence,label in zip(sentences,labels):
        if label in small_label:
            sample_sentences.append(sentence)
            sample_labels.append(label)
    with open(os.path.join(dataname,"{}-{}.tsv".format(des,mode)),"w") as f:
        f.write("sentence label\n")
        for sentence,label in zip(sample_sentences, sample_labels):
            f.write(sentence + " ")
            f.write(label)
            f.write("\n")
        
def count_dataset(datadir,dataset,mode = "train"):
    dataname = os.path.join(datadir,dataset)
    
    with open(os.path.join(dataname,  "{}.tsv".format(mode)),"r") as f:
        data = f.readlines()
    data = data[1:]
    sentences = []
    labels = []
    d = defaultdict(list)
    for l in data:
        t = l.strip().split()
        sentences.append(" ".join(t[:-1]))
        
        labels.append(t[-1])
        d[t[-1]].append(1)
    for k,v in d.items():
        print("label:{} number:{}".format(k,len(v)))



datadir = "../data"

dataset = "IMDB"

des = "split"

count_dataset(datadir,dataset,mode = "train")
# split_dataset(datadir,dataset,mode="dev",des=des)
