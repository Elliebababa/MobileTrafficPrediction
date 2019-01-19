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
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, LSTM, ConvLSTM2D, Input, Dropout, TimeDistributed, Activation, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.regularizers import l2
import keras.backend as K
K.set_image_data_format('channels_first')
#utils for training
from utils import *
import metrics

#define parameters for training 
days_test = 7
T = 144
len_test = T * days_test
lr = 0.0002
look_back = 6 # look back of interval, the same as len_closeness
lstm_latent_dim = 32

nb_epoch = 500
patience = 10  # early stopping patience
nb_epoch_cont = 0
batch_size = 1

#model for training
modelbase = 'gcnmodel'

def lstm(lookback = 6,dimension = 1):
    inputX = Input(shape = (lookback,dimension))
    lstm = LSTM(lstm_latent_dim)(inputX)
    y = Dense(1, activation = 'tanh')(lstm)
    model = Model(input = inputX, output = y)
    return model


def load_data_(file_name = '../../data/processed/internet_t10_s3030_4070.h5', look_back = 6, days_test = 7, T = 144):
    #load data for convlstm cnnlsrm fclstm
    f = h5py.File(file_name, 'r')
    data = f['data'].value
    n_slots, n_features, n_rows, n_cols = data.shape
    data = data.reshape(n_slots, n_rows*n_cols)
    return data.T

def makedataset(data,lookback = 6):
    X = []
    Y = []
    for i in range(look_back+1, data.shape[0]):
        X.append(data[i-look_back-1:i-1,])
        Y.append(data[i,])
    X = np.asarray(X).reshape((len(X),lookback,1))
    Y = np.asarray(Y)
    return X,Y

def main():
    #getmodel
    model = lstm()
    adam = Adam(lr = lr)
    model.compile(loss = 'mse', optimizer = adam, metrics = [metrics.rmse, metrics.mape, metrics.ma])
    all_scores_train = []
    all_scores_test = []
    #model.summary()
    #get data
    data = load_data_() #nodes x slots
    ts1 = time.time()
    
    trueY_train = []
    predictY_train = []
    trueY_test = []
    predictY_test =[]

    for i in range(len(data)):
        print('grid %d.....'%(i))
        ts = time.time()
        #makedata set
        testslots = T * days_test
        trainx,trainy = makedataset(data[i,:-testslots])
        testx,testy = makedataset(data[i,-testslots:])
        print('trainx shape:',(trainx.shape))
        print('trainy shape:',(trainy.shape))
        print('testx shape:',(testx.shape))
        print('testy shape:',(testy.shape))

        #scaler
        print(trainy,testy)
        mmn = MinMaxScaler(feature_range=(-1,1))
        trainlen = len(trainy)
        Y = np.concatenate([trainy, testy],axis = 0)
        Y = mmn.fit_transform(Y.reshape(-1,1))
        trainy,testy = Y[:trainlen],Y[trainlen:]
        print(trainy.shape,testy.shape)
        #train
        adam = Adam(lr = lr)
        model.compile(loss = 'mse', optimizer = adam, metrics = [metrics.rmse, metrics.mape, metrics.ma])
        early_stopping = EarlyStopping(monitor='val_rmse', patience=patience, mode='min')
        history = model.fit(trainx, trainy, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping], verbose=0)        
        #evalute
        predict_y_train = model.predict([trainx], batch_size=batch_size, verbose=0)[:,0:1]
        score = model.evaluate(trainx, trainy, batch_size = batch_size, verbose = 0)
        print('Train score: %.6f rmse (norm): %.6f  rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. , score[1] * (mmn._max - mmn._min)/2./ mmn.inverse_transform(np.mean(y_train)), score[2], score[3]))
        predict_y_test = model.predict([testx], batch_size=batch_size, verbose=0)[:,0:1]
        score = model.evaluate(testx, testy, batch_size = batch_size, verbose = 0)
        print('Test score: %.6f rmse (norm): %.6f  rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. , score[1] * (mmn._max - mmn._min)/2./ mmn.inverse_transform(np.mean(y_test)), score[2], score[3]))

        predictY_train.append(mmn.inverse_transform(predict_y_train).reshape(-1).tolist())
        predictY_test.append(mmn.inverse_transform(predict_y_test).reshape(-1).tolist())
        trueY_train.append(mmn.inverse_transform(trainy).reshape(-1).tolist())
        trueY_test.append(mmn.inverse_transform(testy).reshape(-1).tolist())
        print("\nestimate on grid%d ,elapsed time (eval): %.3f seconds\n" % (i,time.time() - ts))
    #all_scores_train = np.asarray(all_scores_train)
    #all_scores_train = np.mean(all_scores_train, axis = 0)
    #all_scores_test = np.asarray(all_scores_test)
    #all_Scores_test = np.mean(all_scores_test,axis = 0)
    print('\n\n')
    evaluate = lambda y1,y2:(metrics.rmse(y1,y2), metrics.rmse(y1,y2)/np.mean(y1),metrics.mape(y1,y2), metrics.ma(y1,y2))
    print('All Train rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f'%(evaluate(trueY_train,predictY_train)))
    print('All Test rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f'%(evaluate(trueY_test,predictY_test)))
    print('elapsed time: %3f seconds\n'%(time.time()-ts1))


    

    
'''
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
    if modelbase == 'lstm':
        model = lstm()
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
    ts = time.wotime()
    history = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, model_checkpoint,tensor_board], verbose=1)
    model.save_weights(os.path.join('MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\ntrain model elapsed time (training): %.3f seconds\n" % (time.time() - ts))
    

    #==== evaluating model ===================================================================================
    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(fname_param)
    score = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f  rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. , score[1] * (mmn._max - mmn._min)/2. / mmn.inverse_transform(np.mean(y_train)), score[2], score[3]))
    score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
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
    score = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. , score[1] * (mmn._max - mmn._min)/2./ mmn.inverse_transform(np.mean(y_train)), score[2], score[3]))

    score = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
    ts = time.time()
    score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f nrmse : %.6f mape: %.6f ma: %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. , score[1] * (mmn._max - mmn._min)/2. / mmn.inverse_transform(np.mean(y_test)), score[2], score[3]))
    print("\nevaluate final model elapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))
    
'''
if __name__ == '__main__':
    main()




    
         
