import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

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

    file2 = [val.split() for val in open(label_dir, 'r').read().splitlines()]
    label_to_idx, idx_to_label = {}, {} 
    for line in file2:
        label_to_idx[line[0]] = int(len(label_to_idx)+1)
        idx_to_label[label_to_idx[line[0]]] = line[0]
    
    label_to_idx['unk'] = len(label_to_idx)+1
    idx_to_label[label_to_idx['unk']] = 'unk'

    return word_to_idx, idx_to_word, label_to_idx, idx_to_label


def make_batch(sent, labels, device):
    word_to_idx, idx_to_word, label_to_idx, idx_to_label = get_vocab() 

    sent_batch = 50    
    sent_trim = sent_batch*math.floor(len(sent)/sent_batch)
    sent = sent[:sent_trim]
    sent = [word_to_idx[word] if word in word_to_idx else word_to_idx['unk'] for word in sent]
    input_sent = torch.tensor([sent[:-sent_batch]], dtype=torch.long).to(device)
    output_sent = torch.tensor([sent[sent_batch:]], dtype=torch.long).to(device)
    #sent_batch = [[sent[idx:idx+sent_batch]] for idx in range(0, len(sent), sent_batch)]
    
    label_batch = 10
    label_trim = label_batch*math.floor(len(labels)/label_batch)
    labels = labels[:label_trim]
    labels = [label_to_idx[word.lower()] if re.search(r'\d+',word)==None else label_to_idx['number'] for word in labels]
    input_labels = torch.tensor([labels[:-label_batch]], dtype=torch.long).to(device)
    output_labels = torch.tensor([labels[label_batch:]], dtype=torch.long).to(device)
    #label_batch = [[labels[idx:idx+label_batch]] for idx in range(0, len(labels), label_batch)]    

    return input_sent.reshape(-1,sent_batch), output_sent.reshape(-1,sent_batch), input_labels.reshape(-1,label_batch), output_labels.reshape(-1,label_batch)   


def train_sent_label_embeds(train_data, sent_len, epochs):
    
    word_to_idx, idx_to_word, label_to_idx, idx_to_label = get_vocab()    
    device = torch.device('cuda:1' if torch.cuda.is_available else 'cpu')
   
    input_sent_pad, target_sent_pad, input_label_pad, output_label_pad = 0, 0, 0, 0
    loss_func = nn.CrossEntropyLoss()

    sentence_model = Transformer(len(word_to_idx), len(word_to_idx), input_sent_pad, target_sent_pad, device).to(device)
    sent_optimizer = optim.SGD(sentence_model.parameters(), lr=0.01)
    label_model = Transformer(len(label_to_idx), len(label_to_idx), input_label_pad, output_label_pad, device).to(device)
    label_optimizer = optim.SGD(label_model.parameters(), lr=0.01)
 
    # ==== Pretraining the sent and label embeddings === # 
    losses = []
    for epoch in range(epochs):
      total_loss = 0
      for idx, line in enumerate(train_data):  
        sentence_model.zero_grad()
        label_model.zero_grad()

        encoder = line[0]
        decoder = line[1]
        sentences = [words for sent, label in encoder for words in sent]
        labels = [words for sent, label in encoder for words in label]
        s_inp, s_out, l_inp, l_out = make_batch(sentences, labels, device)
        
        s_transformer = sentence_model(s_inp, s_out).to(device)
        s_transformer = s_transformer.reshape(-1,len(word_to_idx))
        s_loss = loss_func(s_transformer, s_out.reshape(-1))
    
        l_transformer = label_model(l_inp, l_out).to(device)
        l_transformer = l_transformer.reshape(-1,len(label_to_idx))
        l_loss = loss_func(l_transformer, l_out.reshape(-1))     
      
        current_loss = s_loss + l_loss
        
        current_loss.backward()
        sent_optimizer.step()
        label_optimizer.step()
        
        total_loss += current_loss.item()
      losses.append(total_loss)
    print(losses)

