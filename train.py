import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dropout, LSTM, Lambda, Dense, Activation, Masking, TimeDistributed, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import random
import pickle
from carddata import read_cards, process_data, get_vocab
from callbacks import HistoryCheckPoint

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #-1 sets to CPU 

#Set GPU to use dynamic memory scaling
physical_devices = tf.config.experimental.list_physical_devices('GPU')
try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    print("Error setting GPU memory config")


def main():
    START_TOKEN = '$'
    END_TOKEN = '#'
    SEQ_LENGTH = 400
    HIDDEN_DIM = 300
    EMBED_DIM = HIDDEN_DIM
    BATCH_SIZE = 32
    LAYERS = 2
    DROPOUT = .2
    EPOCHS = 25
    TEST_SPLIT = 0.1
    #FOLDS = 5
    PAD = 'right'
    START_EPOCH = 5 #actually counts from this plus 1
    
    datafile = 'encoded.txt'
    data = read_cards(datafile, START_TOKEN, END_TOKEN)

    vocab = get_vocab(data, pad=True, pad_token=None)
    int_to_char = {i:char for i, char in enumerate(vocab)}
    char_to_int = {char:i for i, char in enumerate(vocab)}
    
    #keep random shuffle order across training loads
    seeded_rand = random.Random("mtg")
    seeded_rand.shuffle(data) 
    
    #Splits
    n_test = int(len(data) * TEST_SPLIT)
    n_test += (len(data)-n_test)%BATCH_SIZE
    test_data = data[-n_test:]
    data = data[:-n_test]
   
   #Note: Original implementation used a generator but ended up reverting for testing and never went back
    X,y = process_data(data, char_to_int, SEQ_LENGTH, len(vocab), pad=PAD)
    test_X, test_y = process_data(test_data, char_to_int, SEQ_LENGTH, len(vocab))
    
    #model = create_model_sequential(SEQ_LENGTH, HIDDEN_DIM, vocab, DROPOUT)
    model = create_model_functional(SEQ_LENGTH, HIDDEN_DIM, vocab, DROPOUT)
    model_directory = 'models/test/'
    #model = load_model(model_directory + 'model')
    
    print(model.summary())
    #print("Printing model architecture")
    #plot_model(model, 'models/EncDec, Len=400/diagram.png', show_shapes=True, show_layer_names=False)
    
    hist_ckpt = HistoryCheckPoint(model_directory + 'history')
    best_ckpt = ModelCheckpoint(model_directory + 'best_epoch={epoch:02d},val_loss={val_loss:.6f}', monitor='val_loss', save_best_only=True)
    curr_ckpt = ModelCheckpoint(model_directory + 'current')
    h = model.fit(X, y, validation_data=(test_X,test_y), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[hist_ckpt, best_ckpt, curr_ckpt]) 
    model.save(model_directory + 'model')
    
    print("Calculating perplexity...")
    perplexity, cross_entropy = model_perplexity(model, test_X, test_y)
    print(cross_entropy)
    print(perplexity)
     


def create_model_sequential(SEQ_LENGTH, HIDDEN_DIM, vocab, DROPOUT):
    #Define model
    model = Sequential()
    model.add(Input(shape=(SEQ_LENGTH,)))
    model.add(Embedding(len(vocab), HIDDEN_DIM, mask_zero=True)) 
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(Dropout(DROPOUT))
    #model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    #model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Dense(len(vocab))))
    model.add(Activation('softmax'))
    #model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop",)# metrics=[perplexity])
    #model.compile(loss=sparse_categorical_crossentropy_masked, optimizer="rmsprop",)
    #model.compile(loss=custom_loss, optimizer="rmsprop",)
    
    return model
    

def create_model_functional(SEQ_LENGTH, HIDDEN_DIM, vocab, DROPOUT):
    input_layer = Input(shape=(SEQ_LENGTH,))
    embed_layer = Embedding(len(vocab), HIDDEN_DIM, mask_zero=True)
    embed1 = embed_layer(input_layer)
    
    lstm1 = LSTM(HIDDEN_DIM, return_sequences=True)(embed1)
    dropout1 = Dropout(DROPOUT)(lstm1)
    lstm2 = LSTM(HIDDEN_DIM, return_sequences=True)(dropout1)
    dropout2 = Dropout(DROPOUT)(lstm2)
    #lstm3 = LSTM(HIDDEN_DIM, return_sequences=True)(dropout2)
    #dropout3 = Dropout(DROPOUT)(lstm3)
    
    last = dropout2
    
    #Normal
    #dense1 = TimeDistributed(Dense(len(vocab)))(last)  
    #softmax1 = Activation('softmax')(dense1)#(weight_layer)#(transpose)
    
    #Weight Tied
    dense1 = Dense(HIDDEN_DIM, activation='relu')(last)
    tied_dense = WeightTiedDense(embed_layer, trainable=False)(dense1)
    #tied_dense = WeightTiedDense(embed_layer, trainable=True)(dense1)
    softmax1 = Activation('softmax')(tied_dense)    
    

    model = Model(input_layer, softmax1)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")
    
    
    return model
    
    
class WeightTiedDense(Layer):
    def __init__(self, target_layer, activation=None, **kwargs):
        self.target_layer = target_layer
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)
        
    def build(self, batch_input_shape):
        #self.biases = self.add_weight(name="bias", shape=[self.target_layer.input_shape[-1]], initializer="zeros")
        super().build(batch_input_shape)
    
    def call(self, inputs):
        result = tf.keras.backend.dot(inputs, tf.keras.backend.transpose(self.target_layer.weights[0]))
        return result
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_layer': self.target_layer,
        })
        return config    
          

#choose uniformly and independently among ~X tokens    
def perplexity_simple(cce):
    return tf.exp(cce).numpy()
    
#finds each sample, then avgs that.  does not weight length of each sample    
def model_perplexity(model, test_X, test_y, mask_zeros=True):       
    tot_cce = 0.0
    for i in range(len(test_y)):
        x = test_X[i]
        x = x[np.newaxis, :]
        y_true = test_y[i]
        y_pred = model.predict(x)

        timesteps = len(y_true)
        if mask_zeros:
            for t in range(len(y_true)):
                if y_true[t] == 0:
                    y_true = y_true[0:t]
                    y_pred = y_pred[:,0:t,:]
                    timesteps = t
                    break
        
        print(i)
        cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
        avg_cce = K.mean(cross_entropy).numpy()
        tot_cce += avg_cce
        #if avgcce > 1.0 or avgcce < 0.0:
        #    print(cross_entropy).numpy()
    cross_entropy = tot_cce / len(test_y)
    perplexity = K.exp(cross_entropy).numpy()
    return perplexity, cross_entropy      
    
    
if __name__ == "__main__":
    main()    