from preprocess import *
from transformer_model import *
from pretrain_embeds import *


def main():
    """
        A = torch.ones([5200, 500]) 
        B = torch.ones([5200],dtype=torch.long)
        print(loss(A,B))
    """
    sent_len, epochs = 30, 10
    train_data = read_data(sent_len, test=True)
    train_sent_label_embeds(train_data, sent_len, epochs)
    
    
if __name__ == "__main__":
    main()
