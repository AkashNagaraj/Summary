from preprocess import *
from transformer_model import *
from build_embeds import *
from cnn import *

import re, argparse
import time


def convert_to_vec(sent, label):
    word_to_idx, idx_to_word, label_to_idx, idx_to_label = get_vocab()
    sent = sent.lower()
    new_sent = re.sub(r'[0-9]+','number', sent)
    new_sent = re.sub(r'[^\w\s]','', sent).strip()
    new_sent = new_sent.split()
    sent_vec = [word_to_idx[word] if word in word_to_idx else word_to_idx['unknown'] for word in new_sent]

    new_label = [re.sub(r'[^\w\s]','',word.lower()) if re.search(r'\d+',word.lower())==None else 'number' for word in label.strip().split()]
    label_vec = [label_to_idx[label] for label in new_label if label!='']
    return sent_vec, label_vec


def make_vector(data):
    line_sent, line_label = [], []
    for sent_, label_ in data:
        sent, label = convert_to_vec(sent_, label_)
        if len(sent)>0:
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


def sample_transformer(data_dir, current_file):
    word_to_idx, _, _ = get_vocab()
    current_file = 'final_'+type_+'.txt'



def transformer_data(current_file):
   
    #current_file = type_ + '.txt'
    data = open(current_file,'r').readlines()
    word_to_idx, _, _, _ = get_vocab()
    file_data = [json.loads(val) for val in data]

    for idx, line in enumerate(file_data): # check
        
        section_data = line['sections']
        section_labels = line['section_names']
        input_data = combine_sections(section_data, section_labels)
        line_sent, line_label = make_vector(input_data) 
                
        section_abstract = line['abstract_text']
        section_abstract = ' '.join(section_abstract)
        section_abstract = re.sub(r'[^\w\s]',' ',section_abstract)
        section_abstract = [word_to_idx[val] if val in word_to_idx else word_to_idx['unknown'] for val in section_abstract.split() if len(val)>1]
        yield (line_sent, line_label), section_abstract


# Combine the sentence and label embedings
def combine_embeds(cuda_num, type_):
    
    data_dir, weights_dir = 'data/pubmed-dataset/', 'data/models/'
    sent_weights = torch.load(weights_dir + 'sentence_model.pth')
    sent_embeddings = sent_weights['encoder.word_embedding.weight']
    label_weights = torch.load(weights_dir+'label_model.pth')
    label_embeddings = label_weights['label_weights']
    
    input_embed_len = len(label_embeddings) + sent_embeddings.shape[1] 
    print("Input embed len:",input_embed_len)

    word_to_idx, idx_to_word, label_to_idx, idx_to_label = get_vocab()
    device = torch.device('cuda:'+cuda_num if torch.cuda.is_available() else 'cpu')
    input_pad, output_pad = 0, 0
    cnn_model = CNN(input_embed_len).to(device)

    # Transfomer with input and ouput size of len(sentence) + len(labels)
    abstract_model = Transformer(len(word_to_idx) + len(label_to_idx), len(word_to_idx) + len(label_to_idx), input_pad, output_pad, device).to(device)
    abstract_optimizer = optim.SGD(abstract_model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    current_file = data_dir + type_ + '.txt'
    sent_count = 0
    total_loss = []

    for index, (encoder_data, output) in enumerate(transformer_data(current_file)):
        
        count, section, label = 0, encoder_data[0], encoder_data[1]
        t_input = torch.empty(size=(len(section), 306)).to(device)
        abstract_model.zero_grad()
        print(len(label))
        # Concat sentence and label embeddings
        for idx, sent in enumerate(section):
            count+=1
            s = torch.tensor(sent, dtype = torch.long).to(device)
            
            # Combine word embeds for sentence
            s_embeds = sent_embeddings[s]
            s_row = torch.tensor(s_embeds.shape[0])
            s_embeds = torch.sum(s_embeds,dim=0).to(device)
            s_embeds = s_embeds/torch.sqrt(s_row).to(device)
            torch.cuda.empty_cache()

            # Combine words embeds for labels
            l_embeds = label_embeddings[sent_count]
            l_embeds = torch.tensor(l_embeds,dtype=torch.long).to(device)
            sent_count +=  1

            #print("Size - s_embeds:{}, l_embeds:{}".format(s_embeds.shape, l_embeds.shape))
            
            # Concatenate label and sent embeds
            line_input = torch.cat((s_embeds.reshape(1,-1), l_embeds), 1).reshape(1,-1)
            t_input[idx] = torch.tensor(line_input, dtype=torch.long)       
        
        t_output = torch.tensor([output], dtype = torch.long) #.to(device)
        
        print("Shape -> t_input:{},{}, t_output:{},{}".format(t_input.shape, t_input.type(), t_output.shape, t_output.type()))
        
        abstract_prediction = abstract_model(t_input, t_output).to(device)
        loss = loss_func(abstract_prediction, t_output.reshape(-1))
        loss.backward()
        abstract_optimizer.step()
        torch.cuda.empty_cache()
        total_loss.append(loss.item())
        
        print("Abstract loss : ",total_loss)
        sys.exit()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda", "--cuda", help="Enter a number")
    args = parser.parse_args()

    """    
    # == Get word embeddings for sentences and labels === #
    sent_len, epochs, test_size = 30, 30, 50
    complete_data = read_data(sent_len, test_size, test_runtime=False) 
    train_data = complete_data['train']
    train_sent_label_embeds(train_data, sent_len, epochs, args.cuda)
    """

    # == Combine them and build input for transformer == #
    combine_embeds(args.cuda, type_='train')
    
    
if __name__ == "__main__":
    main()
