import torch
import torch.nn as nn
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
    label_to_idx, idx_to_label = {'unknown':0}, {0:'unknown'}
    for line in file2:
        label_to_idx[line[0]] = int(len(label_to_idx)+1)
        idx_to_label[label_to_idx[line[0]]] = line[0]

    return word_to_idx, idx_to_word, label_to_idx, idx_to_label


def make_batch(sent, labels):
    word_to_idx, idx_to_word, label_to_idx, idx_to_label = get_vocab() 

    sent_batch = 50    
    sent_trim = sent_batch*math.floor(len(sent)/sent_batch)
    sent = sent[:sent_trim]
    sent = [word_to_idx[word] if word in word_to_idx else word_to_idx['unknown'] for word in sent]
    input_sent = torch.tensor([sent[:-sent_batch]], dtype=torch.long)
    output_sent = torch.tensor([sent[sent_batch:]], dtype=torch.long)
    #sent_batch = [[sent[idx:idx+sent_batch]] for idx in range(0, len(sent), sent_batch)]
    
    label_batch = 10
    label_trim = label_batch*math.floor(len(labels)/label_batch)
    labels = labels[:label_trim]
    labels = [label_to_idx[word.lower()] if re.search(r'\d+',word)==None else label_to_idx['number'] for word in labels]
    input_labels = torch.tensor([labels[:-label_batch]], dtype=torch.long)
    output_labels = torch.tensor([labels[label_batch:]], dtype=torch.long)
    #label_batch = [[labels[idx:idx+label_batch]] for idx in range(0, len(labels), label_batch)]    

    return input_sent.reshape(-1,sent_batch), output_sent.reshape(-1,sent_batch), input_labels.reshape(-1,label_batch), output_labels.reshape(-1,label_batch)    


class Embedding_Creation(nn.Module):
    def __init__(self, vocab_size, vocab_context, label_size, embed_dim):
        super(Embedding_Creation, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.word_embeds = nn.Embedding(vocab_size, embed_dim, scale_grad_by_freq=True)
        self.label_embeds = nn.Embedding(label_size, embed_dim, scale_grad_by_freq=True)
        self.vocab_linear = nn.Linear(embed_dim*vocab_context, 100)
        self.label_linear = nn.Linear(embed_dim, 100) 

    def forward(self,sent,label):
        sent_out = self.word_embeds(sent).view((1,-1))
        sent_out = F.relu(self.vocab_linear(sent_out))
        label_out = self.label_embeds(label).view((1,-1))
        label_out = F.relu(self.label_linear(label_out))
        return sent_out, label_out

def train_transformer(train_data, sent_len):
    
    word_to_idx, idx_to_word, label_to_idx, idx_to_label = get_vocab()    
    device = torch.device('cpu') #device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    input_sent_pad, target_sent_pad = 0, 0
    sentence_model = Transformer(len(word_to_idx), len(word_to_idx), input_sent_pad, target_sent_pad, device).to(device)


    for line in train_data:

        encoder = line[0]
        decoder = line[1]
        sentences = [words for sent, label in encoder for words in sent]
        labels = [words for sent, label in encoder for words in label]
        s_inp, s_out, l_inp, l_out = make_batch(sentences, labels)
        
        out = sentence_model(s_inp, s_out).to(device)
        print(out.shape)
        
        """
        for idx in range(0, len(sent_batch)-1):
            input_ = torch.tensor(sent_batch[idx], dtype=torch.long).to(device)
            output = torch.tensor(sent_batch[idx+1], dtype=torch.long).to(device)
            out = sentence_model(input_, output)
            print(out.shape)
        """


def main():
    sent_len = 30

    train_data = read_data(sent_len, test=True)
    train_transformer(train_data, sent_len)
    
    
if __name__ == "__main__":
    main()
