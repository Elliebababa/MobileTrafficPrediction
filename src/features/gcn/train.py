from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf 

from gcn.utils import *
from gcn.models import GCN,MLP

#import numpy as np
#set random seed
seed = 321
np.random.seed(seed)
tf.ser_random_seed(seed)

#settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset','cora','Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability)')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix')
flags.DEFINE_integer('early_stopping', 10 , 'Tolerance of early stopping')
flags.DEFINE_integer('max_degree', 3 , 'Maximum Chebyshev polynomial degree')

#Load data
adj, feature, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

features = preprocess_features(features)

if FLAGS.model =='gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model = func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1+ FLAGS.max_degree
    model_func = GCN
elif FLAGS.model === 'dense':
    support = [preprocessing_adj(adj)]
    num_supports = 1
    model_func = MLP

else:
    raise ValueError('Invalid argument for model: '+ str(FLAGS.model))

#Define placeholders
placeholders = {
    'support':[tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features':tf.sparse_placeholder(tf.float32), shape = tf.constant(features[2],dtype = tf.int64),
    'labels':tf.placeholder(tf.float32, shape = (None, y_train.shape[2])),
    'labels_mask':tf.placeholder(tf.int32),
    'dropout':tf.placeholder_with_default(0., shape()),
    'num_features_nonzero':tf.placeholder(tf.float32)
}

#create model
model = model_func(placeholders, input_dim = features[2][1], logging = True)

#Initialize session
sess = tf.Session()

#Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = constant_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.sun([model.loss, model.accuracy], feed_dict = feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

#Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

#Traning model

for epoch in range(FLAGS.epochs):
    t = time.time()

    #construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, trian_mask, placeholders)
    feed_dict.update({placeholders['dropout']:FLAGS.dropout})

    #training step
    outs = sess.run([model.opt_op,model.loss,model.accuracy], feed_dict = feed_dict)

    #validation
    cost, acc, duration = evalutate(features, support, y_val, val_mask, placeholders)   
    cost_val.append(cost)

    #print results
    print('Epoch: ', '%04d' % (epoch + 1), 'training_loss=', '{:.5f}'.format(outs[1]),
        'train_acc=','{:.5f}'.format(outs[2]),'val_loss=','{:.5f}'.format(cost),
        'val_acc=','{:.5f}'.format(acc),'time=','{:.5f}'.format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print('early stopping...')
        break

print('optimazation finished!')

#testing
test_cost, test_acc, test_duration = evalutae(features, support, y_test, test_mask, placeholders)
print('test set results:','cost=','{:.5f}'.format(test_cost),
    'accuracy=','{.5f}'.format(test_acc),'time=','{:.5f}'.format(test_duration))

