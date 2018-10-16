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
from scipy.sparse.linalg.eigen.arpack import eigsh

def list_of_csr_to_sparse_tensor(adjs):
    # a little bug here
    # I saved the csr matrix in the shape of 10001 * 10001
    # and I convert back to 10000 * 10000 csr sparse tensor
    # sorry for the careless operation...
    indices = []
    data = []
    shape0 = len(adjs)
    shape1,shape2 = adjs[0].shape
    shape1 -= 1
    shape2 -= 1
    for i in range(shape0):
        adj = adjs[i].toarray()
        adj = scipy.sparse.coo_matrix(adj[1:,1:])
        row = adj.row
        col = adj.col
        idx = np.ones(row.shape, dtype = np.int32)*i
        indx = list(zip(idx, row, col))
        indices += indx
        data += list(adj.data)
    if indices == []:
        indices = [(0,0,0)]
        data = [0]
    #indices = np.transpose(indices)
    #return tf.SparseTensor(indices = indices, values = data, dense_shape = (shape0,shape1,shape2))
    return indices,data#,[shape0,shape1,shape2]

#help to return the UTC time with given time interval
def showTime(x):
    tt = time.gmtime(x/1000)
    t = '{}.{} {}:{}'.format(tt.tm_mon,tt.tm_mday,tt.tm_hour,tt.tm_min)
    return t


#form the Internet usage of each timeInterval
#target matrix is naturally the next timeInterval
def dataMatrix(timeInt,data,nodes):
    #generate the dataMatrix of Internet over the whole grids of a given timeInt
    #data is a pandas dataframe with hierachical index of [grid, time]
    M = nodes
    m = int(np.sqrt(M))
    X = np.zeros((M+1,))
    for i in range(1,M+1):
        if timeInt in data.loc[i].index:
            X[i] = data.loc[i].loc[timeInt]['Internet']
    X = X[1:].reshape((m,m))
    return X


#preview the internet stength of several time interval
def plotInternet(seriesList, nrows, ncols):
    assert len(seriesList) == nrows*ncols
    fig, axes = plt.subplots(nrows, ncols, figsize = (15,3*nrows))
    M = 10000
    m = int(np.sqrt(M))
    n = nrows * ncols
    for i,ax in enumerate(axes.flat):
        x = dataMatrix(seriesList[i],d)
        im = ax.imshow(x,vmin = 0)
        #ax.set_title('Interval{}'.format(seriesList[i]))
        ax.set_title(showTime(seriesList[i]))
        
    fig.subplots_adjust(right = 0.8)
    cax = fig.add_axes([0.82,0.16,0.02,0.7])
    fig.colorbar(im, cax = cax)
    plt.show()

#show the pairs that have interaction with each other
def showNonzero(interaction):
    print('d = |V| = {},k|V| < |E| = {}'.format(10000,interaction.nnz))
    plt.spy(interaction,markersize = 2, color = 'blue')

def showDistribution(interaction):
    plt.boxplot(interaction.data,sym = 'r+')
    print('variance: ',np.var(interaction.data))
    print('mean: ',np.mean(interaction.data))

def make_dataset(str_time,end_time,data,dir_,nodes = 10000,inter = 600000):
    #return data_all data_all
    print('making dataset from {}'.format(dir_))
    data_x = []
    data_adj = []
    for timeInterval in range(str_time,end_time+1,inter):
        num = inter // 600000
        
        datax = np.zeros((nodes,1))
        adj = scipy.sparse.csr_matrix((nodes+1,nodes+1))

        for i in range(num):
            datax += dataMatrix(timeInterval+i*600000,data,nodes).reshape(nodes,1)
            
            fp_ = '{}/{}.npz'.format(dir_,timeInterval+i*600000)

            if os.path.exists(fp_):
                adj_tmp = load_npz(fp_)
            else:
                adj_tmp = scipy.sparse.csr_matrix((nodes+1,nodes+1))

            adj = adj.toarray()[0:nodes+1][0:nodes+1]
            adj = scipy.sparse.csr_matrix(adj)
            adj += adj_tmp

        adj = adj_to_sym(adj)
        adj = laplacian2(adj)
        data_x.append(datax)
        data_adj.append(adj)

    print('finish preparing the dataset....')
    return list(zip(data_x,data_adj))


def laplacian1(W, normalized = True,sparse = True):
        
         # Degree matrix.
    d = W.sum(axis=0)

        # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

        # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

    
def laplacian2(adj, normalized = True,sparse = True):
       
    #normalize
    adj = adj.tocoo()
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    #scaled
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adj.shape[0])

    return scaled_laplacian

def adj_to_sym(interaction):
    adj = interaction.toarray()
    adj = adj + adj.T
    return scipy.sparse.csr_matrix(adj)