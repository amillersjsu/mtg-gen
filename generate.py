import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from carddata import read_cards, get_vocab, process_data

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #-1 sets to CPU 

#Set GPU to use dynamic memory scaling
physical_devices = tf.config.experimental.list_physical_devices('GPU')
try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    print("Error setting GPU memory config")

#Couldn't find exactly where I got this, but it seems to be floating around on quite a few sources
#Not implemented by me.
#Higher temp causes a more uniform distribution
#Lower temp causes higher probabilities to dominate more
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)   
    
    
def generate_text(model, char_to_int, int_to_char, SEQ_LENGTH, seed='', temperature=1.0):
    output = seed
    ix = [char_to_int[c] for c in output]
    X = np.zeros((1, SEQ_LENGTH))
    i = 0
    while output[-1] != '#' and len(output) < SEQ_LENGTH:
        X[0] = np.roll(X[0], -1) #this makes the function very slow. consider rewriting
        X[0, -1] = ix[-1]
        preds = model.predict(X)
        pred = sample(preds[0][-1], temperature)
        ix.append(pred)
        c = int_to_char[pred]
        output.append(c)
        i += 1
    text = ''.join(output)    
    return text    
    
    
def main():
    START_TOKEN = '$'
    END_TOKEN = '#'
    SEQ_LENGTH = 400
    PAD = 'right'
    TEMPERATURE = 1.0
    
    datafile = 'encoded.txt'
    data = read_cards(datafile, START_TOKEN, END_TOKEN)
    vocab = get_vocab(data, pad=True, pad_token=None)
    int_to_char = {i:char for i, char in enumerate(vocab)}
    char_to_int = {char:i for i, char in enumerate(vocab)}
    
    model = load_model('models/test/model')
    print(model.summary())
    
    user_input = ''
    while user_input == '':
        seed = ['$']
        text = generate_text(model, char_to_int, int_to_char, SEQ_LENGTH, seed=seed, temperature=TEMPERATURE)
        print(text)
        user_input = input()

    
    
if __name__ == "__main__":
    main()    