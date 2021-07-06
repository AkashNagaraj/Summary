import json
import math
import itertools


def preprocess_data(sentence, max_len):
    words = sentence.split()
    if len(words)>max_len:
        sentence = ' '.join(words[:max_len])
    else:
        add = ' unkwown ' * (max_len-len(words))
        sentence += add # Find out what to add
    return sentence.split()


def make_input(data, labels):
    input_data, min_len, max_len = [], 6, 25
    for idx, list_ in enumerate(data):
        for sent in list_:
            if len(sent)>min_len:
                input_data.append((preprocess_data(sent, max_len),labels[idx].split()))  # Currently len is fixed try variable
    return input_data


def build_train(file_data):
    
    """
    {
    'article_text':[[S1,S2,..],[S1,S2..],...,[S1,S2..]]
    'abstract_text':[[S1],[S1],...,[S1]]
    'section_names': [[S1,S2..],[S1,S2,..],...,[S1,S2,..]]
    'sections': [ [[S1,S2,..],[],..,[]], [[S1,S2,..],[],..,[]], ..., [[S1,S2,..],[],..,[]]] 
    'article_id':
    }
    """
    
    train_data = []
    decoder_len = 300

    for line in file_data:
        #content = line['article_text']
        section_data = line['sections']
        section_labels = line['section_names']
        encoder_data = make_input(section_data, section_labels) 
        
        decoder_data = ' '.join([sent for sent in line['abstract_text']]).split() 
        # Use this for decoder of same length - preprocess_data(' '.join([sent for sent in line['abstract_text']]),decoder_len)
       
        train_data.append((encoder_data, decoder_data))

    return train_data


def read_data(test):
    data_dir = "../data/pubmed-dataset/"
    current = "val.txt" # Change later to train
    read_lines = open(data_dir+current,'r').readlines()
    data = [json.loads(val) for val in read_lines] 
    
    if test:
        train_data = build_train(data[:10])
    else:
        train_data = build_train(data)

    return train_data
    
