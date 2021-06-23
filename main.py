
def main():
    data = get_data()
    transfomers(data) # Use one sent to generate all others...
    lstm_fb() # Concatenate hidden layers[f,b] of sentence embedding and add it to decoder
    word_cnn() # Multiple for single sent and concatenate and add to decoder
    token_emb() # Add grammar/syntax/topic/style... something
    eval_score() # Rouge?


if __name__ == "__main__":
    main()
