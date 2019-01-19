from __future__ import print_function
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from keras.layers import Input, Dropout, Add, TimeDistributed, LSTM, Dense
from keras.models import Model 
from keras import activations, initializers
from keras.engine import Layer
import keras.backend as K
import numpy as np
import tensorflow as tf
from utils import *
import scipy.sparse as sp
FILTER = 'chebyshev'
MAX_DEGREE = 2

class GraphConvolution(Layer):
    def __init__(self,units,support = 1, graphfilter = FILTER,  activation = 'relu',use_bias = True, kernel_initializer = 'glorot_uniform',**kwargs):
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
        self.A = np.load('/home/hyf/MobileTrafficPrediction/src/models/kegra/wight_matrix_lb_weekly_4070_2.npy')
        #self.A = np.load('/home/hyf/MobileTrafficPrediction/src/models/kegra/wight_matrix_lb_weekly_4070.npy')
        if self.filter == 'localpool':
            print('using local...')
            #self.A = normalize_adj(self.A)
            adj = sp.coo_matrix(self.A)
            print('building diags..')
            d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
            print('normalizing..')
            a_norm = adj.dot(d).transpose().dot(d)
            print('csr to array..')
            self.A = a_norm.toarray()
            #discard far neighbor(those with larger weight)
                #self.A[np.where(self.A < 0.9)] = 0
            #self.A = np.load('wight_matrix_lb_weekly_4070.npy')
            self.A = tf.convert_to_tensor(self.A,dtype = 'float32')
            self.basis.append(self.A)
        elif self.filter =='chebyshev':
            print('using chebyshev...')
            #L = normalized_laplacian(self.A)
            adj = sp.coo_matrix(self.A)
            d = sp.diags(np.power(np.array(adj.sum(1)),-0.5).flatten(),0)
            a_norm = adj.dot(d).transpose().dot(d)
            laplacian =  sp.eye(adj.shape[0]) - a_norm
            #L_scaled = rescale_laplacian(L)
            try:
                print('calculating largest eigenvalue of normalized graph laplacian...')
                largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors = False)[0]
            except ArpackNoConvergence:
                print('Eigenvalue calculation did not converge! Using largest_eigval = 2 instead.')
                largest_eigval = 2
            X = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0]) #L_SCALED
            #T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
            print('calculating chebyshev polynomials up to order {}...'.format(MAX_DEGREE))
            T_k = list()
            T_k.append(sp.eye(X.shape[0]).tocsr())
            T_k.append(X)

            def chebyshev_recurrence(T_k_1, T_k_2,X):
                X_ = sp.csr_matrix(X,copy = True)
                return 2*X_.dot(T_k_1) - T_k_2
            
            for i in range(2, MAX_DEGREE+1):
                T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2],X))

            self.support = MAX_DEGREE + 1
            self.basis = [tf.convert_to_tensor(i.toarray(),dtype = 'float32') for i in T_k]


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
    #wrapped = TimeDistributed(GraphConvolution(nb_nodes, support))(wrapped)
    #pass graph convolutional layers as list of tensors
    #Y = GraphConvolution(nb_nodes, support)(X_in)
    lstm = LSTM(64)(wrapped)
    Y = Dense(nb_nodes, activation = 'tanh')(lstm)
    model = Model(inputs = X_in, outputs = Y, name = 'gcnlstm')
    return model
