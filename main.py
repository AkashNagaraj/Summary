import torch
import torch.nn as nn
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def train(): 
    """
        transfomers(data) # Use one sent to generate all others...
        lstm_fb() # Concatenate hidden layers[f,b] of sentence embedding and add it to decoder
        word_cnn() # Multiple for single sent and concatenate and add to decoder
        token_emb() # Add grammar/syntax/topic/style... something
    """
    train_iter = WikiText2(split='train')
    print(train_iter)


def main():
    train()
    #eval_score() # Rouge?


if __name__ == "__main__":
    main()
