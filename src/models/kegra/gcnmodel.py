from __future__ import print_function
from keras.layers import Input, Dropout, Add, TimeDistributed, LSTM, Dense
from keras.models import Model 
from keras import activations, initializers
from keras.engine import Layer
import keras.backend as K
import numpy as np
import tensorflow as tf
#FILTER = 'localpool'
MAX_DEGREE = 2

class GraphConvolution(Layer):
    def __init__(self,units,support = 1, graphfilter = 'localpool',  activation = 'tanh',use_bias = True, kernel_initializer = 'glorot_uniform',**kwargs):
        super(GraphConvolution,self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get('zeros')
        self.support = support # support means the inner filter
        self.filter = graphfilter
        self.basis = list()
        self.buildbasis()

    def buildbasis(self):
        if self.filter == 'localpool':
            self.A = np.load('/home/hyf/MobileTrafficPrediction/src/models/kegra/wight_matrix_lb_weekly_4070.npy')
            #self.A = np.load('wight_matrix_lb_weekly_4070.npy')
            self.A = tf.convert_to_tensor(self.A,dtype = 'float32')
            self.basis.append(self.A)


    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape,self.units)
        return output_shape

    def build(self, input_shapes):
        input_dim = input_shapes[1]
        self.kernel = self.add_weight(shape = (input_dim * self.support, self.units), initializer = self.kernel_initializer, name = 'kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape = (self.units,), initializer = self.bias_initializer, name = 'bias')
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        print('input : ', inputs )
        features = inputs[0]
        print('features shape : ',features.shape)
        print('basis shape : ',self.basis[0].shape)
        supports = list()
        for i in range(self.support):
            print(i)
            supports.append(K.dot(features,self.basis[i]))
        supports = K.concatenate(supports, axis = 0)
        supports = K.transpose(supports)
        print('supports shape : ', supports.shape)
        print('kernel shape : ', self.kernel.shape)
        output = K.dot(supports, self.kernel)
        if self.bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def buildmodel(input_shape):

    support = 1
    #SHAPE[1] is the channel/ dimension of the data
    print('build model got input shape:',input_shape)
    lookback, dimension, nb_nodes = input_shape
    X_in = Input(shape = (lookback, dimension, nb_nodes))
    wrapped = TimeDistributed(GraphConvolution(nb_nodes, support))(X_in)
    #pass graph convolutional layers as list of tensors
    #Y = GraphConvolution(nb_nodes, support)(X_in)
    lstm = LSTM(64)(wrapped)
    Y = Dense(nb_nodes, activation = 'tanh')(lstm)
    model = Model(inputs = X_in, outputs = Y, name = 'gcnlstm')
    return model
