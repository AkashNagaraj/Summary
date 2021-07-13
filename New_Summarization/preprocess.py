import json
import re
import math
import itertools


def preprocess_data(sentence, max_len):
    words = sentence.split()
    if len(words)>max_len:
        sentence = ' '.join(words[:max_len])
    else:
        add = ' unknown ' * (max_len-len(words))
        sentence += add # Find out what to add

    # Check len of sent print(len(sentence.split()))
    return sentence.split()


def make_input(data, labels, max_len):
    input_data, min_len= [], 6
    for idx, list_ in enumerate(data):
        for sent in list_:
            data = (preprocess_data(sent, max_len),labels[idx].split())
            input_data.append(data)  # Currently len is fixed try variable
    return input_data


def build_train(file_data, sent_len):
    
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
    line_labels = []

    for line in file_data:
        #content = line['article_text']
        section_data = line['sections']
        section_labels = line['section_names']
        line_labels.append(section_labels)
        encoder_data = make_input(section_data, section_labels, sent_len)         
        decoder_data = ' '.join([sent for sent in line['abstract_text']]).split() 

        # Use this for decoder of same length - preprocess_data(' '.join([sent for sent in line['abstract_text']]),decoder_len)
        train_data.append((encoder_data, decoder_data))

    total_labels = list(set([word.lower() if re.search(r'\d+',word)==None else 'number' for lines in line_labels for labels in lines for word in labels.split()]))
    
    label_dir = 'data/pubmed-dataset/labels'
    file_ = open(label_dir,'w')
    for val in total_labels:
        file_.write(val+"\n")

    return train_data


def read_data(max_sent_len, test):
    data_dir = "data/pubmed-dataset/"
    current = "val.txt" # Change later to train
    read_lines = open(data_dir+current,'r').readlines()
    data = [json.loads(val) for val in read_lines] 
    
    if test:
        train_data = build_train(data[:10], max_sent_len)
    else:
        train_data = build_train(data, max_sent_len)

    return train_data
    
