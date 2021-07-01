import torch
import torch.nn as nn

from preprocess import *
from transformer_model import *

def get_vocab():
    word_idx_dir = "../data/pubmed-dataset/vocab"
    file = [val.split() for val in open(word_idx_dir, 'r').read().splitlines()]
    word_to_idx, idx_to_word = {}, {}
    for line in file:
        word_to_idx[line[0]]=int(line[1])
        idx_to_word[int(line[1])]=line[0]
    return word_to_idx, idx_to_word


def convert_vector(data, size):
    data = " ".join([val for val in data]).split()
    if len(data)>size:
        data = data[:size]
    else:
        data += ["unknown"] * (size-len(data))

    word_to_idx, _ = get_vocab()
    new_data = []
    for val in data:
        if val in word_to_idx:
            new_data.append(word_to_idx[val])
        else:
            new_data.append(word_to_idx['unknown'])
        
    #new_data = [word_to_idx[val] if val in word_to_idx else word_to_idx['unknown'] for val in data]
    return new_data


def get_batch(input, target, num_lines):
    input_size, target_size = 200, 500
   
    min_seq_len = min([len(input[i]) for i in range(0, num_lines)])
    #print(min_seq_len)

    for i in range(0, num_lines):
        for seq in input[i]:
            new_input = (convert_vector(seq, input_size))
            new_input = torch.tensor(new_input, dtype=torch.long)    
        
        new_target = convert_vector(target[i], target_size)
        new_target = torch.tensor(new_target, dtype=torch.long)
        print(new_input.shape)
        print(new_target.shape)


def train(data, num_lines, device):
    
    # Hyperparameters 
    src_pad_idx = 0
    trg_pad_idx = 0
    word_to_idx, idx_to_word = get_vocab()
    src_vocab_size = max(idx_to_word.keys())
    trg_vocab_size = max(idx_to_word.keys())
    print(src_vocab_size)
    input_size, target_size = 200, 500
    epochs = 1

    # Transformer 1
    model1 = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)  
    input_ = data['article_text']
    target = data['sections']

    #get_batch(input_, target_, num_lines)

    for i in range(epochs):
        for i in range(0, num_lines):
            for seq in input_[i]:
                new_input = (convert_vector(seq, input_size))
            new_target = (convert_vector(target[i], target_size))
            #print(new_target)
        """
                new_input = torch.tensor((convert_vector(seq, input_size))).reshape(10,-1).to(device)
                new_target = torch.tensor(convert_vector(target[i], target_size)).reshape(10,-1).to(device)
                print(new_input.shape, new_target.shape)
                out = model1(new_input, new_target)
                print(out.shape)
        """

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, size = read_data(test=True) # dict_keys(['article_id', 'article_text', 'abstract_text', 'labels', 'section_names', 'sections']
    train(data, size, device)
    """
    Transformer 1 : inp = article, out1 = sent
    Transformer 2 : inp = out1*sent, out2 = abstract
    """


if __name__ == "__main__":
    main()
