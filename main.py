from preprocess import *
from transformer_model import *
from build_embeds import *
from cnn import *

import re
import argparse


def convert_to_vec(sent, label):
    
    word_to_idx, idx_to_word, label_to_idx, idx_to_label = get_vocab()
    sent = sent.lower()
    new_sent = re.sub(r'[^\w\s]','', sent)
    new_sent = new_sent.strip().split()
    sent_vec = [word_to_idx[word] if word in word_to_idx else word_to_idx['unknown'] for word in new_sent]

    new_label = [re.sub(r'[^\w\s]','',word.lower()) if re.search(r'\d+',word.lower())==None else 'number' for word in label.strip().split()]
    label_vec = [label_to_idx[label] for label in new_label]
    return sent_vec, label_vec


def make_vector(data):
    line_sent, line_label = [], []
    for sent_, label_ in data:
        sent, label = convert_to_vec(sent_, label_)
        if len(sent)>0:
            #print("Sent val: {}".format(len(sent)))
            line_sent.append(sent)
            line_label.append(label)
    return line_sent, line_label


def combine_sections(section_data, section_labels):
    input_data = [] 
    for idx, list_ in enumerate(section_data):
        new_sent = []
        for sent in list_:
            new_sent.append(sent)
        new_sent = ' '.join(new_sent)
        input_data.append((new_sent, section_labels[idx]))
    return input_data


def transformer_data(data):
    
    word_to_idx, _, _, _ = get_vocab()
    file_data = [json.loads(val) for val in data]

    for idx, line in enumerate(file_data[:5]): # check
        
        section_data = line['sections']
        section_labels = line['section_names']
        input_data = combine_sections(section_data, section_labels)
        line_sent, line_label = make_vector(input_data) 
                
        section_abstract = line['abstract_text']
        section_abstract = ' '.join(section_abstract)
        section_abstract = re.sub(r'[^\w\s]',' ',section_abstract)
        section_abstract = [word_to_idx[val] if val in word_to_idx else word_to_idx['unknown'] for val in section_abstract.split() if len(val)>1]
        yield (line_sent, line_label), section_abstract


def combine_embeds(cuda_num, type_):
    
    data_dir, weights_dir = 'data/pubmed-dataset/', 'data/models/'

    sent_weights = torch.load(weights_dir + 'sentence_model.pth')
    sent_embeddings = sent_weights['encoder.word_embedding.weight']
    label_weights = torch.load(weights_dir+'label_model.pth')
    label_embeddings = label_weights['encoder.word_embedding.weight']
    
    input_embed_len = label_embeddings.shape[1] + sent_embeddings.shape[1] 
    
    word_to_idx, idx_to_word, label_to_idx, idx_to_label = get_vocab()
    device = torch.device('cuda:'+cuda_num if torch.cuda.is_available() else 'cpu')
    input_pad, output_pad = 0, 0
    cnn_model = CNN(input_embed_len).to(device)
    abstract_model = Transformer(len(word_to_idx) + len(label_to_idx), len(word_to_idx) + len(label_to_idx), input_pad, output_pad, device).to(device)
    loss_func = nn.CrossEntropyLoss()

    current_file = type_+'.txt'
    read_lines = open(data_dir+current_file,'r').readlines()

    for index, (input_, output) in enumerate(transformer_data(read_lines)):

        count, sent, label = 0, input_[0], input_[1]
        transformer_input = torch.empty(size=(len(sent), input_embed_len)).to(device)
        abstract_model.zero_grad()

        for idx, val in enumerate(sent):
            count+=1
            s = torch.tensor(val, dtype = torch.long).to(device)
            l = torch.tensor(label[idx], dtype = torch.long).to(device)
           
            # Combine word embeds for sentence
            s_embeds = sent_embeddings[s]
            s_row = torch.tensor(s_embeds.shape[0])
            s_embeds = torch.prod(s_embeds,dim=0).to(device)
            s_embeds = s_embeds/torch.sqrt(s_row).to(device)
    
            # Combine words embeds for labels
            l_embeds = label_embeddings[l]
            l_row = torch.tensor(l_embeds.shape[0])
            l_embeds = torch.prod(l_embeds,dim=0).reshape(1,-1)/torch.sqrt(l_row)
               
            line_input = torch.cat((s_embeds, l_embeds), 0).reshape(1,-1).to(device)
            transformer_input[idx] = line_input        
        
        transformer_input = torch.unsqueeze(transformer_input,0).unsqueeze(0)
        transformer_input = torch.tensor(cnn_model.forward(transformer_input), dtype=torch.long).reshape(1,-1).to(device)

        transformer_output = torch.tensor([output], dtype = torch.long).to(device)

        #print("Input shape:{}, input type :{}, output shape:{}, output type:{}".format(transformer_input.shape, type(transformer_input), transformer_output.shape, type(transformer_output)))
        value = abstract_model(transformer_input, transformer_output).to(device)
        print(value.shape)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda", "--cuda", help="Enter a number")
    args = parser.parse_args()

    # == Get word embeddings for sentences and labels === #
    sent_len, epochs = 30, 10
    train_data, test_data, val_data = read_data(sent_len, test_runtime=True)
    train_sent_label_embeds(train_data, sent_len, epochs, args.cuda)
    
    combine_embeds(args.cuda, type_='train')
    #bleu_check()
    
    
if __name__ == "__main__":
    main()
