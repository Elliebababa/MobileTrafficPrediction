# import numpy as np
from keras import backend as K


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# aliases
mse = MSE = mean_squared_error
# rmse = RMSE = root_mean_square_error

def mape(y_true, y_pred):
    return K.mean((y_true - y_pred)/y_true)

def ma(y_true, y_pred):
    #mean accuracy
    return 1 - mape(y_true, y_pred)

def nrmse(y_true, y_pred):
    return rmse(y_true, y_pred)/K.mean(y_true)

def masked_mean_squared_error(y_true, y_pred):
    idx = (y_true > 1e-6).nonzero()
    return K.mean(K.square(y_pred[idx] - y_true[idx]))


def masked_rmse(y_true, y_pred):
    return masked_mean_squared_error(y_true, y_pred) ** 0.5
