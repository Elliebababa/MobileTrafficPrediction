from __future__ import print_function
import pandas as pd

'''
from tensorflow import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Input, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
'''
#from graph import GraphConvolution
#from kegra.utils import *

import time

from DTW_calculator import DTWdistance,gen_dtw_matrix


#related function
def load_data(file_name = 'grid1.csv',path = '../../../data/processed/gridTraffic30/gridTraffic30'):
    print('loading data from {}...'.format(path))
    grid = pd.read_csv('{}/{}'.format(path,file_name))    
    features = grid.iloc[:,1:-1]
    y = grid.iloc[:,-1]
    return features[:-1],y[1:],y

def build_graph(grids,mode = None):
    num = len(grids)
    if mode == None:
        return np.eye(num)
    elif mode == 'DTW':
        return gen_dtw_matrix(grids,num)

def get_split(y):
    num = len(y)
    train_num = int(num//3*2)
    idx_train = range(train_num)
    idx_test = range(train_num,num)
    y_train = np.zeros(y.shape, dtype = np.float32)
    y_test = np.zeros(y.shape,dtype = np.float32)
    y_train[idx_train] = y[idx_train]
    y_test[idx_test] = y[idx_test]
    return y_train, y_test,idx_train, idx_val
    

# Define parameters

FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience
gridNum = 1

# Get data
X = []
y = []
grids = []
for i in range(1,gridNum+1):
    tmpX,tmpy,gridtraffic = load_data(file_name = 'grid{}.csv'.format(i))
    X.append(tmpX)
    y = y+tmpy
    grids.append(gridtraffic)

print(X,y,grids)

y_train, y_test, idx_train, idx_test = get_splits(y)
'''
# Normalize X
X /= X.sum(1).reshape(-1, 1)


if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM)
    support = 1
    graph = [X, A_]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

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

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

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
    model.fit(graph, y_train, sample_weight=train_mask,
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




    
         
