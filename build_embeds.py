import ast, math, time, sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, IterableDataset

from itertools import islice
from preprocess import *
from transformer_model import *


def get_vocab():
    word_dir = "data/pubmed-dataset/vocab"
    label_dir = "data/pubmed-dataset/labels"
    
    file1 = [val.split() for val in open(word_dir, 'r').read().splitlines()]
    word_to_idx, idx_to_word = {}, {}
    for line in file1:
        word_to_idx[line[0]] = int(len(word_to_idx)+1)
        idx_to_word[word_to_idx[line[0]]] = line[0]

    file2 = [val.split() for val in open(label_dir, 'r').read().strip().splitlines() if val.split()!=[]]
    label_to_idx, idx_to_label = {}, {} 
    for line in file2:
        try:
            label_to_idx[line[0]] = int(len(label_to_idx)+1)
            idx_to_label[label_to_idx[line[0]]] = line[0]
        except:
            print("Error in line :",line)
            break;

    label_to_idx['unk'] = len(label_to_idx)+1
    idx_to_label[label_to_idx['unk']] = 'unk'

    return word_to_idx, idx_to_word, label_to_idx, idx_to_label


def make_tensor(data, type_, device, percent):
    
    if type_ =='sent':
        word_to_idx, idx_to_word, _, _ = get_vocab() 
    else:
        _, _, word_to_idx, idx_to_word = get_vocab()

    length = len(data)
    size = math.floor((length*percent)*100)
    new_data = [word_to_idx[word] if word in word_to_idx else word_to_idx['unk'] for word in data]
    
    # Normalize the word data
    word_data = [val for val in idx_to_word.keys()]
    avg = sum(word_data)/len(word_data)
    var = np.var(word_data)
    new_data = [(val-avg)/var for val in new_data]

    new_tensor = torch.tensor([new_data[:size]], dtype=torch.long).to(device)

    return new_tensor.reshape(-1,length)   


def combine_list(list_data):
    check = list_data[0]
    new_list = [check]
    for val in list_data:
        if val!=check:
            new_list.append(val)
            check=val
    new_list = [word for sub in new_list for word in sub]
    return new_list


def convert_data(data):
    
    new_label, new_sentence = [], []
    for line in data[0]: # 0 has the encoder value
        label, sent = [], []

        for section in line[0]:
            if section[0]!=[] and section[1]!=[]:
                label.append(section[1])
                sent.append(section[0])
        
        if label!=[] and sent!=[]:
            new_label.append(combine_list(label))
            new_sentence.append(combine_list(sent))
        
    return new_label, new_sentence


def train_sent(sentence_data, cuda_num, epochs):
    
    word_to_idx, idx_to_word, label_to_idx, idx_to_label = get_vocab()    
    device = torch.device('cuda:' + cuda_num if torch.cuda.is_available else 'cpu')
   
    input_sent_pad, target_sent_pad, input_label_pad, output_label_pad = 0, 0, 0, 0
    
    loss_func = nn.CrossEntropyLoss()
    sentence_model = Transformer(len(word_to_idx), len(word_to_idx), input_sent_pad, target_sent_pad, device).to(device)
    sent_optimizer = optim.SGD(sentence_model.parameters(), lr=0.01)
 
    # ==== Pretraining the sent and label embeddings === # 
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        start = time.time()    
        
        for idx, sent in enumerate(sentence_data):
            sentence_model.zero_grad()
            
            s_inp = make_tensor(sent[:int(len(sent)/2)], 'sent', device, 100) # Use 30%
            s_out = make_tensor(sent[int(len(sent)/2):], 'sent', device, 100) # Use 30%
            
            try:
                s_transformer = sentence_model(s_inp, s_out).to(device)
                s_transformer = s_transformer.reshape(-1,len(word_to_idx))
                s_loss = loss_func(s_transformer, s_out.reshape(-1))
            except:
                print(" Large Size Input : {}, Large Size Output : {}".format(s_inp.size(), s_out.size()))
                sys.exit()

            current_loss = s_loss 
            current_loss.backward()
            sent_optimizer.step()
            torch.cuda.empty_cache()
            total_loss += current_loss.item()
        
        losses.append(total_loss)
        end = time.time()

    torch.save(sentence_model.state_dict(), 'data/models/'+'sentence_model.pth')

def train_labels(data):
    label_dict = {}
    for val in open("data/glove.6B.50d.txt").readlines():
        val = val[:-1]
        word = val.split()[0]
        emb = val.split()[1:]
        label_dict[word] = list(map(float, emb))
    
    complete_tensors = []
    for idx, line in enumerate(data): # Slice to build only fixed number of label embeddings
        if idx==1:
            print(len([[label_dict[word]] if word in label_dict.keys() else [label_dict['unknown']] for word in line]))
        line_tensor = torch.tensor([[label_dict[word]] if word in label_dict.keys() else [label_dict['unknown']] for word in line], dtype=torch.long)
        label_tensor = torch.sum(line_tensor, dim=0)/len(line)
        complete_tensors.append(label_tensor)
    torch.save({"label_weights":complete_tensors}, 'data/models/'+'label_model.pth')        
        

# Pretraining to get some embeddings for the sentence and labels seperately
def train_sent_label_embeds(training_data, sent_len, epochs, cuda_num):
    
    BATCH_SIZE = 10
    training_data = training_data[:(len(training_data)//BATCH_SIZE)*BATCH_SIZE]
    label_data, sentence_data = convert_data(training_data)
    
    train_sent(sentence_data, cuda_num, epochs)    
    train_labels(label_data)    

    #print("Data from build embeddings : ", len(training_data))

