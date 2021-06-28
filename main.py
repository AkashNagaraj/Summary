import torch
import torch.nn as nn

import json
import math
import itertools


def split_data(data):
    train_data = []
    for i,val in enumerate(data):
        train_data.append((data[i],  data[:i] + data[i+1:]))
    


def prepare_data(lines):
    seq_len = 10
    for line in lines:
        line = line + ['unknown'] * (math.ceil(len(line)/seq_len)*seq_len - len(line))
        data = [line[idx:idx+seq_len] for idx in range(0,len(line),seq_len)]
        split_data(data) 


def convert_to_tensor(data):
    #dict_keys(['article_id', 'article_text', 'abstract_text', 'labels', 'section_names', 'sections'])

    article_sent = [' '.join([sent for sent in line['article_text']]).split() for line in data]
    abstract_sent = [' '.join([sent for sent in line['abstract_text']]).split() for line in data]
    prepare_data(article_sent)


def read_data():
    data_dir = "../data/pubmed-dataset/"
    
    current = "val.txt" # Change later to train
    read_lines = open(data_dir+current,'r').readlines()
    read_lines = [json.loads(val) for val in read_lines] 
    
    test = 3
    return read_lines[:test] # Using only first 10 for now


def main():
    data = read_data()
    convert_to_tensor(data)
    """
    train()
    transfomers(data) # Use one sent to generate all others...
    lstm_fb() # Concatenate hidden layers[f,b] of sentence embedding and add it to decoder
    word_cnn() # Multiple for single sent and concatenate and add to decoder
    token_emb() # Add grammar/syntax/topic/style... something
    eval()
    """


if __name__ == "__main__":
    main()
