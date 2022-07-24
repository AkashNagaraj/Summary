import pickle 
import sys, os
import json, re
import math, itertools


def preprocess_labels(labels):
    new_labels = ' '.join([re.sub(r'[^\w\s]','',word.lower()) for word in labels.split()])
    new_labels = re.sub(r'[0-9]+','', new_labels)
    return new_labels.strip()


def preprocess_data(sentence, max_len):
    new_sentence = re.sub(r'[0-9]+','number', sentence)
    new_sentence = re.sub(r'[^\w\s]','',new_sentence.lower()).strip()
    words = new_sentence.split()
    if len(words)>max_len:
        new_sentence = ' '.join(words[:max_len])
        #else:
        # add = ' unknown ' * (max_len-len(words))
        # sentence += add # Find out what to add
    return new_sentence.split()


# Adding labels with each of the section sentences
def make_input(data, labels, max_len):
    input_data, min_len = [], 6
    for idx, list_ in enumerate(data):
        labels[idx] = preprocess_labels(labels[idx])
        for sent in list_:
            new_data = (preprocess_data(sent, max_len),labels[idx].split())
            input_data.append(new_data)  # Currently len is fixed try variable
    return input_data


def build_data(file_data, sent_len, type_):
    
    """
    Json structure - 
    {
    'article_text':[[S1,S2,..],[S1,S2..],...,[S1,S2..]]
    'abstract_text':[[S1],[S1],...,[S1]]
    'section_names': [[S1,S2..],[S1,S2,..],...,[S1,S2,..]] ## labels
    'sections': [ [[S1,S2,..],[],..,[]], [[S1,S2,..],[],..,[]], ..., [[S1,S2,..],[],..,[]]] 
    'article_id':
    }
    """
    
    train_data = []
    decoder_len = 300
    line_labels = []

    # The goal is to combine the [sections + section_names] data which is used as encoder and the abstract text is the decoder data for the transformer.
    for line in file_data:
        #content = line['article_text']
        section_data = line['sections']
        section_labels = line['section_names']
        line_labels.append(section_labels)
        encoder_data = make_input(section_data, section_labels, sent_len)         
        decoder_data = ' '.join(line['abstract_text']).split() #same length - preprocess_data(' '.join(line['abstract_text']),decoder_len)
        train_data.append((encoder_data, decoder_data))
    
    return train_data


def write_word_data(max_sent_len, size, type_, test_runtime):
    data_dir = "data/pubmed-dataset/"             
    current = type_+'.txt'             
    read_lines = open(data_dir+current,'r').readlines()
    train_data = [json.loads(val) for val in read_lines]
    train_data = [train_data[i:i+size] for i in range(len(train_data))[::size]]
    
    # The length of the train_data is the number of sections being used
    if test_runtime:
        train_data = train_data[:10]
    final_data = [build_data(val, max_sent_len, type_) for val in train_data]
    
    return final_data


def read_data(max_sent_len, size, test_runtime):
    data = {}
    types = ['train', 'test', 'val']
    for t in types :
        data[t] = write_word_data(max_sent_len, size, t, test_runtime)
   
    return data
