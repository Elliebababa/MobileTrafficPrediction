from __future__ import print_function

#keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, LSTM, ConvLSTM2D, Input, Dropout, TimeDistributed, Activation, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K

#sequential models

def seq_lstm(input_shape, latent_dim = 64):
    print('cnnlstm selected... compiling...')
    look_back, dimenson = input_shape
    inputX = Input(shape=(look_back, dimenson))
    lstm = LSTM(latent_dim)(inputX)
    outputY = Dense(1, activation='tanh')(lstm)
    model = Model(input = inputX, output = outputY, name ='seq_lstm')
    return model

def seq_cnnlstm(input_shape):
    print('cnnlstm selected... compiling...')
    look_back, dimenson, nb_nodes = input_shape
    rows = int(math.sqrt(nb_nodes))
    input = Input(shape=(look_back, dimenson,  nb_nodes))
    input1 = Reshape(target_shape = (look_back,dimenson, rows, rows))(input)
    wrapped = TimeDistributed(Conv2D(filters = 1,kernel_size = (5,5), strides=(1, 1), padding="same"))(input1)
    input1 = Reshape(target_shape = (look_back, dimenson*nb_nodes))(wrapped)
    lstm = LSTM(64)(input1)
    main_output = Dense(nb_nodes, activation='tanh')(lstm)
    model = Model(input = input, output =main_output, name ='cnnlstm')
    
    return model


