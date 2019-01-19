import pandas as pd
import numpy as np
import h5py as h5
import time
from joblib import Parallel, delayed
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings("ignore")

intervals = 144
daytest = 7
testlen = intervals*daytest
stepahead = 48
f1 = h5.File('../../../data/processed/internet_t10_s3030_4070.h5','r')
dataX = f1['data'].value
dataX = np.squeeze(dataX)
dataX = np.reshape(dataX,(8928,900))
dataX = dataX.T
cfg = (3,1,2)#(p,d,q)

def nrmse(testx, predictions):
    #notice that testx is array while predictions is list of array
    #we should first convert list of array into array
    if type(predictions) == list:
        predictions = np.asarray(predictions).squeeze()
    assert predictions.shape == testx.shape
    meant = np.mean(testx)
    mse = np.mean(np.power(testx-predictions,2))
    rmse = np.sqrt(mse)
    nrmse = rmse/meant
    return nrmse

def arima_forecast(history, config):
    model = ARIMA(history, order = config)
    model_fit = model.fit(transparams = False,disp=0)
    yhat = model_fit.forecast()[0]
    return yhat

def walk_forward_validation(X, config, stepahead = 1):
    predictions = list()
    trainx, testx = X[:-testlen], X[-testlen:]
    history = [x for x in trainx]
    for i in range(0,len(testx),stepahead):
        for step in range(1,stepahead+1):
            print('i:%d, step:%d'%(i,step))
            yhat = arima_forecast(history, config)
            predictions.append(yhat)
            history.append(yhat)
        history[-stepahead:] = testx[i:i+stepahead]
    nrmse_score = nrmse(testx, predictions)
    return nrmse_score

def score_model(data, cfg, stepahead = 1, debug = False):
    gridNum,X = data
    result = None
    if debug:
        result = walk_forward_validation(X,cfg,stepahead)
    else:
        try:
            result = walk_forward_validation(X,cfg,stepahead)
        except:
            result = None
    if result is not None:# and gridNum % 100 == 0:
            print('gridNum:%d, nrmse:%.3f'%(gridNum,result))
    return(gridNum, result)



def arima_test_all(dataX, cfg, stepahead = 1,parallel = True, n_jobs = 32):
    gridnums = len(dataX)
    print('testing on %d grids, with p = %d, d = %d, q = %d'%(gridnums,cfg[0], cfg[1], cfg[2]))
    print('forecast %d step ahead...'%(stepahead))
    scores = None
    if parallel:
        print('using parallel..')
        executor = Parallel(n_jobs = n_jobs, backend = 'multiprocessing')
        tasks = (delayed(score_model)(data,cfg,stepahead) for data in enumerate(dataX))
        scores = executor(tasks)
    else:
        scores = list()
        for data in enumerate(dataX):
            nrmse = score_model(data,cfg,stepahead)
            scores.append(data[0],nrmse)
    scores = [r for r in scores if r[1]!= None]
    nrmses = list(list(zip(*scores))[1])
    print('finish all...')
    print('mean nrmses: %.3f'%(np.mean(nrmses)))
    return scores

if __name__ == '__main__':
    print('ARIMA TESTING...')
    print(dataX)
    scores = arima_test_all(dataX,cfg,stepahead)

