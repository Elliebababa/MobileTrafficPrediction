from __future__ import print_function

#numerical package
import pandas as pd
from scipy import sparse
#system package
import click
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
import metrics
#model
from kegra import convlstm
#from deepst import deepst_model

from kegra import gcnmodel


# define parameters for graph convolution

#define parameters for training 
days_test = 7
T = 144
len_test = T * days_test
lr = 0.0002
look_back = 6 # look back of interval, the same as len_closeness

nb_epoch = 500
patience = 1  # early stopping patience
nb_epoch_cont = 100
batch_size = 1

#training file path
path_result = 'RESULT'
path_model = 'MODEL'
CACHEDATA = False
DATAPATH = './'
path_cache = os.path.join(DATAPATH,'CACHE')

#model for training
modelbase = 'gcnmodel'

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

def build_model(modelbase = 'convlstm',input_shape = (6, 1, 900)):
    print('using model...',modelbase)
    if modelbase == 'convlstm':
        model = convlstm.convlstm(input_shape)
    if modelbase == 'cnnlstm':
        model = convlstm.cnnlstm(input_shape)
    if modelbase == 'fclstm':
        model = convlstm.fclstm(input_shape)
    if modelbase == 'deepst':
        model = deepst_model.builddeepst()
    if modelbase == 'gcnmodel':
        model = gcnmodel.buildmodel(input_shape)
    adam = Adam(lr = lr)
    model.compile(loss = 'mse', optimizer = adam, metrics = [metrics.rmse, metrics.mape, metrics.ma])
    model.summary()
    
    #from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file='model_{}.png'.format(modelbase), show_shapes = True)
    
    return model


import click
@click.command()
@click.option('--modelbase',default='convlstm',help='convlstm,cnnlstm,fclstm,deepst')
def main(modelbase):
    modelbase = modelbase
    # Get data
    print('loading data...')
    ts = time.time()
    if modelbase == 'deepst':
        #fname = os.path.join(DATAPATH, 'CACHE', 'C{}_P{}_T{}.h5'.format(len_closeness, len_period, len_trend))
        X_train, y_train, X_test, y_test, mmn, timestamps_train, timestamps_test = deepst_model.load_data(preprocess_name='preprocessing.pkl')
        #cache(fname, X_train, Y_train, X_test, Y_test, 0, timestamps_train, timestamps_test)
    else:
        #convlstm, cnnlstm, fclstm, deepst, gcnlstm
        X_train, X_test, y_train, y_test, mmn = load_data_1()
        print(" X_train shape : {} \n X_test shape : {} \n y_train shape: {} \n y_test shape : {}".format(np.asarray(X_train).shape, np.asarray(X_test).shape, np.asarray(y_train).shape, np.asarray(y_test).shape))
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        """   if modelbase == 'gcnmodel':
                           #A = np.asarray(A)
                           X_train = np.sum(X_train, axis = 1)
                           print(X_train.shape)
                           X_train = [X_train,np.tile(A,(X_train.shape[0],1))]
                           X_test = [X_test, np.tile(A,(X_test.shape[0],1))]
                           print('type: ',type(X_train), type(A))"""

    #print("\n days (test): ", [v[:8] for v in timestamps_test[0::T]])
    print("\n elapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    #==== compiling model ==================================================================================    
    
    print('=' * 10)
    print("compiling model...")
    ts = time.time()
    model = build_model(modelbase, X_train[0].shape)
    hyperparams_name = 'model_{}'.format(modelbase)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
    print(hyperparams_name)
    print(fname_param)
    early_stopping = EarlyStopping(monitor='val_rmse', patience=patience, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
    tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 0, write_graph = True, write_images = False, embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)
    print("\ncompile model elapsed time (compiling model): %.3f seconds\n" %(time.time() - ts))
    
    #==== training model ===================================================================================
    print('=' * 10)
    print("training model...")
    ts = time.time()
    history = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, model_checkpoint,tensor_board], verbose=1)
    model.save_weights(os.path.join('MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\ntrain model elapsed time (training): %.3f seconds\n" % (time.time() - ts))
    

    #==== evaluating model ===================================================================================
    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(fname_param)
    score = model.evaluate(X_train, y_train, batch_size=y_train.shape[0], verbose=0)
    print('Train score: %.6f rmse (norm): %.6f  rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. , score[1] * (mmn._max - mmn._min)/2. / mmn.inverse_transform(np.mean(y_train)), score[2], score[3]))
    score = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. , score[1] * (mmn._max - mmn._min)/2. / mmn.inverse_transform(np.mean(y_test)), score[2], score[3]))
    print("\nevaluate model elapsed time (eval): %.3f seconds\n" % (time.time() - ts))
    
    #==== continue to train model ==============================================================================
    print('=' * 10)
    print("training model (cont)...")
    ts = time.time()
    fname_param = os.path.join('MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, y_train, epochs=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[model_checkpoint, tensor_board])
    pickle.dump((history.history), open(os.path.join(path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join('MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    print("\ncont train model elapsed time (training cont): %.3f seconds\n" % (time.time() - ts))
    
    #==== evaluate on the final model ===============================================================================
    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(X_train, y_train, batch_size=y_train.shape[0], verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. , score[1] * (mmn._max - mmn._min)/2./ mmn.inverse_transform(np.mean(y_train)), score[2], score[3]))

    score = model.evaluate(X_train, y_train, batch_size=y_train.shape[0], verbose=0)
    ts = time.time()
    score = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. , score[1] * (mmn._max - mmn._min)/2. / mmn.inverse_transform(np.mean(y_test)), score[2], score[3]))
    print("\nevaluate final model elapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))
    

if __name__ == '__main__':
    main()




    
         
