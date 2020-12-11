import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pickle

#integrate mtgencode3 in here if necessary

def read_cards(filename, start_token='', end_token=''):
    with open(filename, 'r', encoding="utf-8") as f:
        lines = [start_token + line.rstrip('\n') + end_token for line in f if line != '\n']
    return lines
    
def get_vocab(data, pad=False, pad_token=None):
    vocab = sorted(list(set((c for card_str in data for c in card_str))))
    if pad:
        if pad_token in vocab:
            vocab.remove(pad_token) 
        vocab.insert(0, pad_token)
    return vocab
    
    
#need to take real iterations and not just one cut
def process_data(data, char_to_int, seq_len, vocab_size, pad='right'):
    n_samples = len(data)
    X = np.zeros((n_samples, seq_len))
    Y = np.zeros((n_samples, seq_len)) 
        
    for i,card in enumerate(data):
        seq = [char_to_int[c] for c in card]
        
        #pad
        if len(seq) < seq_len+1: 
            n_pads = seq_len + 1 - len(seq)
            if pad == 'right':
                seq.extend([0] * n_pads)
            elif pad == 'left':
                seq = [0]*n_pads + seq
        
        for timestep in range(seq_len):
            X[i][timestep] = seq[timestep]
            Y[i][timestep] = seq[timestep+1]
     
    return X,Y  


def save_data(X, Y, vocab, vocab_to_int):
    np.save('data/vocab_to_int.npy', vocab_to_int)
    vocab = {i:vocab[i] for i in range(len(vocab))}  #change to dict so np will save properly
    np.save('data/vocab.npy', vocab) #might need to be dict also
    np.save('data/X.npy', X)
    np.save('data/Y.npy', Y)
    

def main():
    START_TOKEN = '$'
    END_TOKEN = '#'

    datafile = 'encoded.txt'
    data = read_cards(datafile, START_TOKEN, END_TOKEN)
    #print(data)
    print("Cards: " + str(len(data)))
    
    seq_len = 400
    avg_len = sum([len(card) for card in data])
    max_len = max([len(card) for card in data])
    min_len = min([len(card) for card in data])
    min_idx = np.argmin([len(card) for card in data])
    max_idx = np.argmax([len(card) for card in data])
    higher = sum([len(card) for card in data if len(card) > seq_len])
    lower = sum([len(card) for card in data if len(card) <= seq_len])
    print("AVG: ", avg_len / len(data))
    print("% of cards over ", seq_len, ":", higher/(higher+lower))
    print("Max: ", max_len)
    print(data[max_idx])
    print("Min: ", min_len)
    print(data[min_idx])
    card_lens = sorted([len(card) for card in data])
    
    n, bins, patches = plt.hist(card_lens,100)
    
    plt.xlabel('Card length')
    plt.ylabel('Number of cards')
    #plt.title('Card Length Histogram')
    plt.show()
    


if __name__ == "__main__":
    main()

