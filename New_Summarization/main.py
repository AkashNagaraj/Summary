from preprocess import *
        

def get_vocab():
    word_idx_dir = "data/pubmed-dataset/vocab"
    file = [val.split() for val in open(word_idx_dir, 'r').read().splitlines()]
    word_to_idx, idx_to_word = {}, {}
    for line in file:
        word_to_idx[line[0]] = int(len(word_to_idx)+1)
        idx_to_word[word_to_idx[line[0]]] = line[0]
    return word_to_idx, idx_to_word


def get_vector(data):
    word_toidx, idx_to_word = get_vocab() 

       
def train_transformer(train_data):
    
    for line in train_data:
        encoder = line[0]
        decoder = line[1]
        for sent, label in encoder:
            sent_embeds = get_vector(sent) 
            label_embeds = 
            combine_embeds = 



def main():
    train_data = read_data(test=True)
    train_transformer(train_data)
    
    
if __name__ == "__main__":
    main()
