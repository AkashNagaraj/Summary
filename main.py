import torch
import torch.nn as nn

from preprocess import *


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


def train(data, size, device):
   
    # Transformer 1 - input:article_text, output:sections
    
    article_input, inp_size = data['article_text'], 90*size  
    section_output, out_size = data['sections'], 500
    
    for i in range(size):
        for idx,sequence in enumerate(article_input[i]):
            input = convert_vector(sequence, inp_size)
            output = convert_vector(section_output[i], out_size)
            input = torch.tensor(input, dtype=torch.long).to(device)
            output = torch.tensor(output, dtype=torch.long).to(device)


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
