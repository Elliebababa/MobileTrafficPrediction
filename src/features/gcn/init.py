import tensorflow as tf
import numpy as np

def uniform(shape, scale= 0.05, name = None):
    '''Uniform init.'''
    initial = tf.random_uniform(shape, minval=-scale,maxval = scale, dtype = tf.float32)
    return tf.Variable(intial,name = name)

