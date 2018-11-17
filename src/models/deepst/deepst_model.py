import os 
import sys 
from os.path import abspath, join , dirname
#print('before',sys.path)
#add module file to sys path
curpath = abspath(dirname('__file__'))
sys.path.insert(0,curpath+'\\deepst')
#print('after',sys.path)

import pickle
import time
from datetime import datetime, timedelta
import numpy as np
import h5py
import click
import pandas as pd
from copy import copy
from keras.optimizers import Adam
import keras.utils



import deepstt
from deepstt.models.STResNet import stresnet

len_closeness = 3
len_period = 1
len_trend = 1
nb_flow = 1
days_test = 7
T = 144#48#144 #interval number of a day

len_test = T * days_test
map_height, map_width = 30,30#100,100#50, 50

def builddeepst(nb_residual_unit = 2, external_dim = None):
    c_conf = (len_closeness, nb_flow, map_height, map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height, map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height, map_width) if len_trend > 0 else None
    print('conf: ',c_conf,p_conf,t_conf)
    model = stresnet(c_conf = c_conf, p_conf = p_conf, t_conf = t_conf, external_dim = external_dim, nb_residual_unit = nb_residual_unit)
    #from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file='model.png', show_shapes = True)
    return model

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


def load_data(preprocess_name = 'preprocessing.pkl'):

    assert(len_closeness + len_period + len_trend > 0)
    data_all = []
    timestamps_all = list()
    fname = '../../data/processed/Nov_internet_data_t10_s3030_4070.h5'
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


def string2timestamp(strings, T=48):
    timestamps = []
    for t in strings:
        timestamps.append(int(t)//1000)
    return timestamps

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
        print('create dataset..',self.T,depends)

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