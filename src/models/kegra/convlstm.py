from __future__ import print_function

#numerical package
import pandas as pd
from scipy import sparse
#system package
import time
import h5py
import os
import math
import pickle
#keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, LSTM, ConvLSTM2D, Input, Dropout, TimeDistributed, Activation, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.regularizers import l2
import keras.backend as K
#utils for training
from utils import *
from metrics import *
#model
from kegra import convlstm


def convlstm(input_shape):
    look_back, dimenson, nb_nodes = input_shape
    rows = int(math.sqrt(nb_nodes))
    input = Input(shape=(look_back, dimenson,  nb_nodes))
    input1 = Reshape(target_shape = (look_back, dimenson, rows, rows))(input)
    conlstm1 = ConvLSTM2D(filters = 1, kernel_size = (3,3), padding="same")(input1)
    main_output = Activation('tanh')(conlstm1)
    main_output1 = Reshape(target_shape = (nb_nodes,))(main_output)
    model = Model(input = input, output = main_output1, name = 'convlstm')
    
    return model

def fclstm(input_shape):
    print('cnnlstm selected... compiling...')
    look_back, dimenson, nb_nodes = input_shape
    rows = int(math.sqrt(nb_nodes))
    input = Input(shape=(look_back, dimenson,  nb_nodes))
    input1 = Reshape(target_shape = (look_back, dimenson*nb_nodes))(input)
    lstm = LSTM(64)(input1)
    main_output = Dense(nb_nodes, activation='tanh')(lstm)
    model = Model(input = input, output =main_output, name ='fclstm')
    
    return model

def cnnlstm(input_shape):
    print('cnnlstm selected... compiling...')
    look_back, dimenson, nb_nodes = input_shape
    rows = int(math.sqrt(nb_nodes))
    input = Input(shape=(look_back, dimenson,  nb_nodes))
    input1 = Reshape(target_shape = (look_back,dimenson, rows, rows))(input)
    wrapped = TimeDistributed(Conv2D(filters = 1,kernel_size = (3,3), strides=(1, 1), padding="same"))(input1)
    input1 = Reshape(target_shape = (look_back, dimenson*nb_nodes))(wrapped)
    lstm = LSTM(64)(input1)
    main_output = Dense(nb_nodes, activation='tanh')(lstm)
    model = Model(input = input, output =main_output, name ='cnnlstm')
    
    return model