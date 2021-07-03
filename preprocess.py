import json
import math
import itertools


def get_data(data, key):
    return [line[key] for line in data]


def get_stats(article, sections):
    avg_word_count = sum([len(sent.split()) for line in sections for list_ in line for sent in list_]) / sum([len(line) for line in sections for list_ in line])
    return avg_word_count


def split_article(data, split):
    new_data = []
    for line in data:
        line_ = [line[idx:idx+split] for idx in range(0, len(line), split)]
        new_data.append(line_)
    return new_data


def preprocess_data(data):
    
    """
    {
    'article_text':[[S1,S2,..],[S1,S2..],...,[S1,S2..]]
    'abstract_text':[[S1],[S1],...,[S1]]
    'section_names': [[S1,S2..],[S1,S2,..],...,[S1,S2,..]]
    'sections': [ [[S1,S2,..],[],..,[]], [[S1,S2,..],[],..,[]], ..., [[S1,S2,..],[],..,[]]] 
    'labels': 
    'article_id':
    }
    """

    article_text = get_data(data, 'article_text')
    abstract_text = get_data(data, 'abstract_text')
    section_names = get_data(data, 'section_names')

    temp_sections, sections = get_data(data, 'sections'), []
    for line in temp_sections:
        temp_sent = [sent for list_ in line for sent in list_]
        sections.append(temp_sent)
  
    word_count = get_stats(article_text, sections)
    train = {'article_text':[], 'abstract_text':[], 'section_names':[], 'sections':[]}
    train['article_text'] = split_article(article_text, split=10) 
    train['abstract_text'] = abstract_text
    train['sections'] = sections # Has detailed info
    train['section_names'] = section_names
    return train


def read_data(test):
    data_dir = "data/pubmed-dataset/"
    current = "val.txt" # Change later to train
    read_lines = open(data_dir+current,'r').readlines()
    data = [json.loads(val) for val in read_lines] 
    if test:
        test = 10 # Use only small data_size
        data = data[:test]
    train_data = preprocess_data(data)
    return train_data, len(data)
