from gcn.init import *
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

#global unique layer ID dictionary for layer name assignent

_LAYER_UIDS = {}

def get_layer_uid(layer_name = ''):
    #helper function , assigns unique layer IDs
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1

    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    #Dropout for sparse tensors
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype = tf.bool)
    pre_out = tf.sparse_retain(x,dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x,y,sparse = False):
    #wrapper for tf.matmul(sparse vs dense)
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x,y)
    else:
        res = tf.matmul(x,y)
    return res

class Layer(object):
    #Base layer class. Defines basic API for all layer objects.
    '''
    Properties
        name: String, defines the variable scope of the layer
        logging: Boolean, switches Tensorflow histogram logging on/off

    Methods
        _call(inputs):Defines computation graph of layer
            (i.e. takes input, returns output)    
        _call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    '''

    def __init__(self, **kwargs):
        allowed_kargs = {'name','logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kargs, 'Invalid keyword argument: '+kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))

        self.name = name
        self.vars = {}

        logging = kwargs.get('logging',False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self,inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name+'/outputs',outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars'+ var, self.vars[var])

class Dense(Layer):
    # Dense layer
    def __init__(self, input_dim)