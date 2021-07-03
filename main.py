import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import *
from transformer_model import *

def get_vocab():
    word_idx_dir = "data/pubmed-dataset/vocab"
    file = [val.split() for val in open(word_idx_dir, 'r').read().splitlines()]
    word_to_idx, idx_to_word = {}, {}
    for line in file:
        word_to_idx[line[0]] = int(len(word_to_idx)+1)
        idx_to_word[word_to_idx[line[0]]] = line[0]
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


def train(data, num_lines, device):
    
    # Hyperparameters 
    src_pad_idx = 0
    trg_pad_idx = 0
    word_to_idx, idx_to_word = get_vocab()
    src_vocab_size = max(idx_to_word.keys())
    trg_vocab_size = max(idx_to_word.keys())
    input_size, target_size = 200, 500
    epochs = 100

    # Transformer 1
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device).to(device)  
    input_ = data['article_text']
    target = data['sections']

    total_loss = []
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    #get_batch(input_, target_, num_lines)

    for i in range(epochs):
      epoch_loss = [] 
      for i in range(0, num_lines):
        line_loss = 0
        for seq in input_[i]:
          model.zero_grad()
          new_input = (convert_vector(seq, input_size))
          new_target = (convert_vector(target[i], target_size))
                
          new_input = torch.tensor([new_input], dtype=torch.long).reshape(5,-1).to(device)
          new_target = torch.tensor([new_target], dtype=torch.long).reshape(5, -1).to(device)
          out = model(new_input, new_target)
          
          t = new_target.shape[0] * new_target.shape[1]
          loss = loss_function(out.view(t,-1), new_target.view(-1))
          loss.backward()
          line_loss += loss.item()
        epoch_loss.append(line_loss/len(input_[i]))
        optimizer.step()
      print(epoch_loss) 

def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    data, size = read_data(test=True) # dict_keys(['article_id', 'article_text', 'abstract_text', 'labels', 'section_names', 'sections']
    train(data, size, device)
    """
    Transformer 1 : inp = article, out1 = sent
    Transformer 2 : inp = out1*sent, out2 = abstract
    """


if __name__ == "__main__":
    main()
