import h5py
import numpy as np
from kegra import buildGraph
class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


def load_data_1(file_name = '../../data/processed/Nov_internet_data_t10_s3030_4070.h5', look_back = 6, days_test = 7, T = 144):
    #load data for convlstm cnnlsrm fclstm 
    f = h5py.File(file_name, 'r')
    data = f['data'].value
    #A = []
    #A = buildGraph.build(np.vstack(data))
    A = np.load('wight_matrix_lb_weekly_4070.npy')
    print('A: ',A)
    print('A shape', A.shape)
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

'''
def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()

def read_cache(fname):
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))

    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test
'''