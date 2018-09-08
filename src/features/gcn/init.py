import tensorflow as tf
import numpy as np

def uniform(shape, scale= 0.05, name = None):
    '''Uniform init.'''
    initial = tf.random_uniform(shape, minval=-scale,maxval = scale, dtype = tf.float32)
    return tf.Variable(intial,name = name)

def glorot(shape, name = None):
    # initialize weights using Glorot's method
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval = -init_range, maxval = init_range, dtype = tf.float32)
    return tf.Variable(initial, name = name)

def zeros(shape, name = None):
    initial = tf.zeros(shape, dtype = tf.float32)
    return tf.Variable(initial, name = name)

def ones(shape, name = None):
    initial = tf.ones(shape, dtype = tf.float32)
    return tf.Variable(initial, name = name)