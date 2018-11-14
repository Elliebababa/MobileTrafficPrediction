from __future__ import print_function
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, LSTM, ConvLSTM2D, Input, Dropout, TimeDistributed, Activation, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.regularizers import l2
from scipy import sparse
from graph_ import GraphConvolution, GraphInput
from utils import *
from metrics import *
import buildGraph
import time
import h5py
import os
import math
import keras.backend as K
import pickle
# Define parameters
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience
days_test = 7
T = 144
lr = 0.0002
look_back = 6 # look back of interval
nb_epoch = 500
nb_epoch_cont = 100
batch_size = 1
path_result = 'RESULT'
path_model = 'MODEL'
CACHEDATA = False
DATAPATH = './'
modelbase = 'cnnlstm'
path_cache = os.path.join(DATAPATH,'CACHE')
  
    
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

def load_data(file_name = '../../../data/processed/Nov_internet_data_t10_s3030_4070.h5'):

    f = h5py.File(file_name, 'r')
    data = f['data'].value
    A = []
    #A = buildGraph.build(np.vstack(data))
    n_slots, n_features, n_rows, n_cols = data.shape
    data = data.reshape(n_slots, n_features, n_rows*n_cols)

    #normalize 
    mmn = MinMaxNormalization()
    # Normalize X
    data = mmn.fit_transform(data)
    
    X = []
    y = []
    for i in range(look_back+1, data.shape[0]):
        X.append(data[i-look_back-1:i-1,])
        y.append(data[i,0])

    slots_test = days_test * T
    x_train = X[:-slots_test]
    x_test = X[-slots_test:]
    y_train = y[:-slots_test]
    y_test = y[-slots_test:] 
    return x_train, x_test, y_train, y_test, mmn, A 

def convlstm(input_shape):
    look_back, dimenson, nb_nodes = input_shape
    rows = int(math.sqrt(nb_nodes))
    input = Input(shape=(look_back, dimenson,  nb_nodes))
    input1 = Reshape(target_shape = (look_back, dimenson, rows, rows))(input)
    conlstm1 = ConvLSTM2D(filters = 1, kernel_size = (3,3), padding="same")(input1)
    main_output = Activation('tanh')(conlstm1)
    main_output1 = Reshape(target_shape = (nb_nodes,))(main_output)
    model = Model(input = input, output = main_output1, name = 'convlstm')
    adam = Adam(lr = lr)
    model.compile(loss = 'mse', optimizer = adam, metrics = [rmse])
    model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model.png', show_shapes = True)
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
    adam = Adam(lr = lr)
    model.compile(loss = 'mse', optimizer = adam, metrics = [rmse])
    model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model.png', show_shapes = True)
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
    adam = Adam(lr = lr)
    model.compile(loss = 'mse', optimizer = adam, metrics = [rmse])
    model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model.png', show_shapes = True)
    return model


def build_model(modelbase = 'convlstm',input_shape = (6, 1, 900)):
    print(modelbase)
    if modelbase == 'convlstm':
        return convlstm(input_shape)
    if modelbase == 'cnnlstm':
        return cnnlstm(input_shape)
    if modelbase == 'fclstm':
        return fclstm(input_shape)

def main():
    # Get data
    print('loading data...')
    ts = time.time()
    X_train, X_test, y_train, y_test, mmn, A = load_data()
    print(" X_train shape : {} \n X_test shape : {} \n y_train shape: {} \n y_test shape : {}".format(np.asarray(X_train).shape, np.asarray(X_test).shape, np.asarray(y_train).shape, np.asarray(y_test).shape))
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    print("\n elapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    #==== compiling model ==================================================================================    
    
    print('=' * 10)
    print("compiling model...")
    ts = time.time()
    model = build_model(modelbase, X_train[0].shape)
    hyperparams_name = 'model_{}'.format(modelbase)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
    tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 0, write_graph = True, write_images = False, embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)
    print("\nelapsed time (compiling model): %.3f seconds\n" %(time.time() - ts))
    
    #==== training model ===================================================================================
    print('=' * 10)
    print("training model...")
    ts = time.time()
    history = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, model_checkpoint,tensor_board], verbose=1)
    model.save_weights(os.path.join('MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))
    

    #==== evaluating model ===================================================================================
    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(fname_param)
    score = model.evaluate(X_train, y_train, batch_size=y_train.shape[0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. ))
    score = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %(score[0], score[1], score[1] * (mmn._max - mmn._min)/2. ))
    print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))
    
    #==== continue to train model ==============================================================================
    print('=' * 10)
    print("training model (cont)...")
    ts = time.time()
    fname_param = os.path.join('MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, y_train, epochs=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[model_checkpoint, tensor_board])
    pickle.dump((history.history), open(os.path.join(path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join('MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))
    #==== evaluate on the final model ===============================================================================
    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(X_train, y_train, batch_size=y_train.shape[0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (score[0], score[1], score[1] * (mmn._max - mmn._min)/2. ))

    score = model.evaluate(X_train, y_train, batch_size=y_train.shape[0] // 48, verbose=0)
    ts = time.time()
    score = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (score[0], score[1], score[1] * (mmn._max - mmn._min)/2. ))
    print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))




'''



if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = sparse.coo_matrix(A)  
    A_ = preprocess_adj(A_, SYM_NORM)
    support = 1
    graph = [X, A_]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=False)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1
    graph = [X]+T_k
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

X_in = Input(shape=(X.shape[1],))
print(X.shape)

#need to do cnn on each period 
#channel first 
def cnn_lstm()：
    input_shape = (5,2,2)
    X_input = Input(input_shape)
    X = TimeDistributed(Conv2D(filters = 64, kernel_size = 1, activation = 'relu'), input_shape = input_shape))(X_input)
    X = TimeDistributed(MaxPooling1D(pool_size = 2))(X)
    X = TimeDistributed(Flatten())(X)
    X = LSTM(50, activation = 'relu')(X)
    X = Dense(1)(X)
    model = Model(inputs = X_input, outputs = X, name = 'ResNet50')
    return model

def cov_lstm():
    passdef


model = cnn_lstm()
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(X,y,epochs = 500, verbose = 0)
x_input = array()


# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu')([H]+G)#, kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
Y = GraphConvolution(1, support, activation='softmax')([H]+G)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, 
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))

'''



if __name__ == '__main__':
    main()




    
         
