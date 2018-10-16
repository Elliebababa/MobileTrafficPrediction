import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse import *
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import collections
import os, time, collections, shutil
#from models import base_model
from helper import *


#debug session
from tensorflow.python import debug as tf_debug


#base model
class base_model(object):
    
    def __init__(self):
        self.regularizers = []
    
    # High-level interface which runs the constructed computational graph.
    
    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            
            batch_data = np.zeros((self.batch_size, data.shape[1]))
            tmp_data = data[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}
            
            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end] = batch_pred[:end-begin]
            
        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions
        
    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)
        #print(predictions)
        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                accuracy, ncorrects, len(labels), f1, loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        return string, accuracy, f1, loss

    def fit(self, train_data, train_labels, val_data, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = train_data[idx,:], train_labels[idx]
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)
            
            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, accuracy, f1, loss = self.evaluate(val_data, val_labels, sess)
                accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                summary.value.add(tag='validation/f1', simple_value=f1)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)
                
                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.
    
    def build_graph(self, M_0):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0), 'data')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        
        self.graph.finalize()
    
    def inference(self, data, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data, dropout)
        return logits
    
    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization
            
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        #path = os.path.dirname(os.path.realpath(__file__))
        try:
            path = os.path.dirname(os.path.abspath(__file__))
        except NameError:  # We are the main py2exe script, not a module
            import sys
            path = os.path.dirname(os.path.abspath(sys.argv[0]))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            #filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            #self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#model
class st_cgcnn(base_model):
    def __init__(self, F, K, M,num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,batch_size=1, eval_frequency=20,
                dir_name=''):
        super().__init__()
        #M denote the number of nodes in the graph
        self.M, self.F, self.K = M,F,K
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.build_graph(M)
    
    def build_graph(self,M):
        self.graph = tf.Graph()
        self.M = M
        with self.graph.as_default():
            #Inputs
            with tf.name_scope('inputs'):
                self.data_x = tf.placeholder(tf.float32, (self.batch_size,M,self.F),'data_x')
                self.data_adj = tf.sparse_placeholder(tf.float32,(self.batch_size,M,M),'data_adj')
                self.data_y = tf.placeholder(tf.float32,(self.batch_size,M),'data_y')
            #Model
            op_logits = self._inference(self.data_x,self.data_adj)
            #p = tf.Print(op_logits,[op_logits],'op_logits')
            self.op_loss,self.op_loss_average = self.loss(op_logits, self.data_y)
            #printout_loss = tf.Print(self.op_loss, [self.op_loss],'loss')
            self.op_train = self.training(self.op_loss, self.learning_rate, self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)
            
            #initialize variables, i.e.. weights and biases
            self.op_init = tf.global_variables_initializer()
            
            #summaries for tensorboard and save for model parameters
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep = 5)
        self.graph.finalize()
        
    def _inference(self,x,adj):
        #Graph convolutional layer
        #x = tf.expand_dims(x, 2) # N x M x F = 1, N is batch size
        t#we use one layer convolution here
        with tf.variable_scope('conv1'):
            with tf.name_scope('filter'):
                x = self.filter_chebyshev(x,adj,1,K = 3) # return N x M x 1
                #p = tf.Print(x,[x],'conv1 filter: ')
        return x   
    

    def filter_chebyshev(self, x, adj, Fout = 1, K = 3):
        #Filtering with chebyshev interpolation
        #data : x of size N x M x F
        #N: number of signals, i.e. number of time interval in our model
        #M: number of vertices
        #F: number of features per signal per vertex
        #N,M,Fin = x.get_shape()
        N,M,Fin = x.get_shape().as_list()
        adj = tf.sparse_tensor_to_dense(adj,default_value = 0)
        #for we have different adj matrix at every time Interval, thus we need to filter each signal seperately
        for i in range(N):
            sig = x[i] #1 x M x Fin
            L = adj[i]
            #to Sparse Tensor
            zero = tf.constant(0,dtype = tf.float32)
            where  = tf.not_equal(L,zero)
            indices = tf.where(where)
            values = tf.gather_nd(L, indices)
            L = tf.SparseTensor(indices, values, L.shape)
            
            #Transform to chebyshev basis
            x0 = sig
            x0 = tf.reshape(x0,[M,Fin]) #M x Fin 
            x = tf.expand_dims(x0,0) # 1 x M x Fin
            def concat(x,x_,d = 0):
                #print(tf.shape(x),tf.shape(x_))
                x_ = tf.expand_dims(x_, d)
                return tf.concat([x,x_],axis = d)
            if K > 1:
                x1 = tf.sparse_tensor_dense_matmul(L,x0)
                x = concat(x,x1)
            for k in range(2,K):
                x2 = 2*tf.sparse_tensor_dense_matmul(L,x1) - x0
                x = concat(x, x2)
                x0,x1 = x1, x2
            x = tf.reshape(x, [K,M,Fin]) # K x M x Fin
            if i == 0:
                x_all = tf.expand_dims(x, -1)
            else:
                x_all = concat(x_all,x,-1)
            
        x_all = tf.transpose(x_all, perm = [3,1,2,0]) #N x M x Fin x K    
        x_all = tf.reshape(x_all, [N*M,Fin*K])
                
        #filter: Fin*Fout filter of order K, i.e. one filterbank per feature pair
        W = self._weight_variable([Fin*K,Fout],regularization = False)
        #printout = tf.Print(W,[W],'W:')
        x_all = tf.matmul(x_all,W) # N*M  x Fout
        return tf.reshape(x_all,[N,M])
    
    def loss(self, values, t_values):
        # adds to the inference model the layers required to generate loss
        with tf.name_scope('loss'):
            with tf.name_scope('mse'):
                mse = tf.square(values - t_values)
                mse = tf.reduce_mean(mse)
            loss = mse
            
        #summaries for tensorboard
        tf.summary.scalar('loss/mse',mse)
        with tf.name_scope('averages'):
            #calculate the average loss up to now
            averages = tf.train.ExponentialMovingAverage(0.9)
            op_averages = averages.apply([mse])
            tf.summary.scalar('loss/avg/total', averages.average(loss))
            with tf.control_dependencies([op_averages]):
                loss_average = tf.identity(averages.average(loss), name = 'control')
        return loss, loss_average
    
    def predict(self, data_x, data_adj, data_y = None, sess = None):
        loss = 0
        size = data_x.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0,size,self.batch_size):
            end = begin + self.batch_size
            end = min([end,size])
            
            batch_data_x = np.zeros((self.batch_size, data_x.shape[1]))
            batch_data_adj = np.zeros((self.batch_size, data_adj.shape[1], data_adj.shape[2]))
            tmp_data_x = data_x[begin:end,:]
            tmp_data_adj = data_adj[begin:end,:]
            #convert sparse matrices
            if type(tmp_data) is not np.ndarray():
                tmp_data = tmp.data.toarray()
            batch_data_x[:end - begin] = tmp_data_x
            batch_data_adj[:end- begin] = tmp_data_adj
            feed_dict = {self.data_x: batch_data_x,self.data_adj:list_of_csr_to_sparse_tensor(batch_data_adj)}
            
            #compute the loss
            if data_y is not None:
                batch_y = np.zeros(self.batch_size)
                batch_y[:end-begin] = data_y[begin:end]
                feed_dict[self.data_y] = batch_y
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss],feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            predictions[begin:end] = batch_pred[:end-begin]
            
        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions
    
    def fit(self, train_data, train_y, val_data, val_y):
        #process time
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph = self.graph)

        #debug sess
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        #notice that x is of array form and adj is the list of csr
        train_x,train_adj = train_data
        val_x,val_adj = val_data
        train_x = np.array(train_x)
        train_adj = list(train_adj)
        val_x = np.array(val_x)
        val_adj = list(val_adj)
        train_y = np.array(train_y)
        val_y = np.array(val_y)
        #print(type(train_x),type(train_adj),type(train_y))
        #train_x,train_adj,train_y = list(train_x),list(train_adj),list(train_y)
        #logging the fit information
        shutil.rmtree(self._get_path('summaries'), ignore_errors = True)
        writer = tf.summary.FileWriter(self._get_path('summaries'),self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors = True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'),'model')
        sess.run(self.op_init)
        #Training
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_x.shape[0]/self.batch_size)
        for step in range(1, num_steps+1):
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_x.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            batch_data_x = train_x[idx,...]
            batch_data_adj = [train_adj[i] for i in idx]
            batch_y = train_y[idx]
            if (len(batch_y.shape) == 3):
                batch_y = batch_y.squeeze(-1)
            pp = list_of_csr_to_sparse_tensor(batch_data_adj)
            feed_dict = {self.data_x : batch_data_x, self.data_adj: pp , self.data_y:batch_y}
            learning_rate,loss_average = sess.run([self.op_train,self.op_loss_average],feed_dict)
            
        #periodical evaluation of the model
            epoch = step * self.batch_size / len(train_x)
            print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
            print('learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
        
            if step % self.eval_frequency == 0 or step == num_steps:
                print("===========================================================")
                #epoch = step * self.batch_size / len(train_x)
                #print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                #print('learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                #string, loss = self.evaluate(val_x,val_adj,val_y,sess)
                #losses.append(mse)
                #print(' validation {}'.format(string))
                print(' time: {:.0f}s(wall{:.0f})s'.format(time.process_time() - t_process, time.time() - t_wall))

                #Summaries for tensorboard
                summary = tf.Summary()
                #summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                #summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                #save model parameters for evaluation
                self.op_saver.save(sess,path,global_step = step)
        #print('validation mse: smallest = {:.2f}, mean={:.2f}'.format(min(losses),np.mean(mse[-10:])))
        
        writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return losses, t_step
            
            
    def evaluate(self, data_x,adj, data_y, sess = None):
        #return mse and loss
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data_x, adj, data_y, sess)
        mse = tf.square(prediction - data_y)
        mse = tf.reduce_mean(mse)
        string = 'mse: {:.2f}'.format(mse)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        return string,mse



#load data
names = ['squareId','timeInterval','countryCode','smsIn','smsOut','callIn','callOut','Internet']
dir_ = '../../data/raw/milan/sms-call-internet-mi/sms-call-internet-mi-Nov/sms-call-internet-mi-2013-11-03.txt'
data = pd.read_table(dir_,names = names)
data = data.fillna(0)
d = data.groupby(['squareId','timeInterval']).sum()
d = d.drop(['countryCode'],axis = 1)
str_time = min(d.index.get_level_values(1))
end_time = max(d.index.get_level_values(1))


#make data set
begin_time = str_time#1383467400000#str_time
period = 600000 * 10
X = make_dataset(begin_time,begin_time+period,d,'../../notebook/demo',6400) # X include internet data and the adj graph
XX = list(list(zip(*X))[0])
ADJ = list(list(zip(*X))[1])
XXX = list(zip(XX,ADJ))
for t,i in enumerate(XX):
    tmp = ((i - np.mean(i))/np.std(i))
    XX[t] = tmp
data_y = XX[1:]

#training

n = len(data_y)
n_train = n//2
n_val = n//2
X_train = XXX[:n_train]
X_val = XXX[n_train:n_train+n_val]
X_test = XXX[n_train+n_val:]

y_train = data_y[:n_train]
y_val = data_y[n_train:n_train+n_val]
y_test = data_y[n_train+n_val:]

#params
myparams = dict()
myparams['num_epochs'] = 1
myparams['batch_size'] = 1
myparams['eval_frequency'] = 5
#architecture
myparams['F'] = 1
myparams['K'] = 3
myparams['M'] = 6400 #output dimensionality of fully connected layers
#Optimization
myparams['learning_rate'] = 1e-5
myparams['decay_rate'] = 0.95
myparams['momentum'] = 0.9
myparams['decay_steps'] = n_train/myparams['batch_size']


#fitting
model =  st_cgcnn(**myparams)
losses, t_step = model.fit(zip(*X_train), y_train, zip(*X_val), y_val)