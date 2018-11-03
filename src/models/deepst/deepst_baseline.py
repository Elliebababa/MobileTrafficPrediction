import os 
import sys 
import pickle
import time
from datetime import datetime, timedelta
import numpy as np
import h5py
import click
import pandas as pd
from copy import copy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.utils

from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics

np.random.seed(1324)

#parameters
nb_epoch = 100
nb_epoch_cont = 20
batch_size = 32
T = 144 #interval number of a day
lr = 0.0002
len_closeness = 18
len_period = 3
len_trend = 3
nb_residual_unit = 2

nb_flow = 1
days_test = 7
len_test = T * days_test
map_height, map_width = 100, 100
datafile ='../../../data/processed/Nov_internet_data_t10_s100100.h5'
path_result = 'RESULT'
path_model = 'MODEL'
CACHEDATA = False
DATAPATH = './'
path_cache = os.path.join(DATAPATH,'CACHE')

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=len_closeness, len_trend=len_trend, TrendInterval=7, len_period=len_period, PeriodInterval=1):
        """current version
        """
        # offset_week = pd.DateOffset(days=7)
        #offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        offset_frame = 24 * 60 // self.T * 60
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness+1),
                   [PeriodInterval * self.T * j for j in range(1, len_period+1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend+1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y

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

class MinMaxNormalization_01(object):
    '''MinMax Normalization --> [0, 1]
       x = (x - min) / (max - min).
    '''
    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * (self._max - self._min) + self._min
        return X

def build_model(external_dim = None):
    c_conf = (len_closeness, nb_flow, map_height, map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height, map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height, map_width) if len_trend > 0 else None
    print('conf: ',c_conf,p_conf,t_conf)
    model = stresnet(c_conf = c_conf, p_conf = p_conf, t_conf = t_conf, external_dim = external_dim, nb_residual_unit = nb_residual_unit)
    adam = Adam(lr = lr)
    model.compile(loss = 'mse', optimizer = adam, metrics = [metrics.rmse])
    model.summary()
    #from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file='model.png', show_shapes = True)
    return model

def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname,'w')
    h5.create_dataset('num', data=len(X_train))
    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' %i, data = data)
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i , data = data)
    h5.create_dataset('Y_train', data = Y_train)
    h5.create_dataset('Y_test', data = Y_test)
    #external_dim = -1 if external_dim or None else int(external_dim)
    h5.create_dataset('T_train', data = timestamp_train)
    h5.create_dataset('T_test', data = timestamp_test)

def stat(fname):
    def get_nb_timeslot(f):
        s = int(f['date'][0])
        e = int(f['date'][-1])
        ts = time.gmtime(s//1000)
        te = time.gmtime(e//1000)
        nb_timeslot = (e - s)/1000 / (24/T * 3600) + T
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    with h5py.File(fname) as f:
        nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / T)
        mmax = f['data'].value.max()
        mmin = f['data'].value.min()
        stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
               '=' * 5 + 'stat' + '=' * 5
        print(stat)

def load_stdata(fname):
    f = h5py.File(fname,'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps

def load_data(preprocess_name = 'preprocessing.pkl'):
    assert(len_closeness + len_period + len_trend > 0)
    data_all = []
    timestamps_all = list()
    fname = datafile
    print('file name: ', fname)
    stat(fname)
    data, timestamps = load_stdata(fname)
    data = data[:, :nb_flow]
    data_all.append(data)
    timestamps_all.append(timestamps)
    print('\n')

    #minmax scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    fpkl = open(preprocess_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    XC, XP, XT = [], [] , []
    Y =[]
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        st = STMatrix(data, timestamps, T)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset()
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y
    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print('XC shape: ', XC.shape, 'XP shape: ', XP.shape, 'XT shape: ', XT.shape, 'Y shape: ', Y.shape)
    print('len_test: ',len_test)
    XC_train, XP_train, XT_train, Y_train = XC[
        :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[
        -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamps_train, timestamps_test = timestamps_Y[
        :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, Y_test.shape)

    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, timestamps_train, timestamps_test


def string2timestamp(strings, T=48):
    timestamps = []
    for t in strings:
        timestamps.append(int(t)//1000)
    return timestamps

def main():
    print('loading data...')
    ts = time.time()
    fname = os.path.join(DATAPATH, 'CACHE', 'C{}_P{}_T{}.h5'.format(
        len_closeness, len_period, len_trend))
    X_train, Y_train, X_test, Y_test, mmn, timestamps_train, timestamps_test = load_data(preprocess_name='preprocessing.pkl')
    cache(fname, X_train, Y_train, X_test, Y_test,
                  0, timestamps_train, timestamps_test)

    print("\n days (test): ", [v[:8] for v in timestamps_test[0::T]])
    print("\n elapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print(
        "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

    ts = time.time()
    model = build_model()
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print("\nelapsed time (compiling model): %.3f seconds\n" %
          (time.time() - ts))

    print('=' * 10)
    print("training model...")
    ts = time.time()
    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) ))
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) ))
    print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("training model (cont)...")
    ts = time.time()
    fname_param = os.path.join(
        'MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
                        model_checkpoint])
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join(
        'MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) ))
    ts = time.time()
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) ))
    print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))



if __name__ == '__main__':
    main()
