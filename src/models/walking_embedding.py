import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse import *
import os

from openne.graph import *
from openne import node2vec

graphPath = '../../data/interim/adj_sparse_10min_2/'

def deepwalk_embedding(time = None, path_length = 10 , num_paths = 5, dim = 10, walkers = 4, window = 5, aggrateNum = 3):

    if time == None:
        return np.zeros(dim)

    #load graph file
    interaction = sp.coo_matrix((10001,10001))
    for i in range(aggrateNum):
        t = time - i*600000
        fp_ =  '{}/{}.npz'.format(graphPath,t)
        if os.path.exists(fp_):
            adj = load_npz(fp_)
            adj = adj.tocoo()
            print(interaction.shape)
            print(adj.shape)
            interaction += adj
    interaction = interaction.tocoo()
    row = interaction.row
    col = interaction.col
    val = interaction.data
    edges = list(zip(row,col,val))
    tmp = 'tmp.txt'
    f = open(tmp, "w+")
    for r,c,v in edges:
        f.write('{} {} {}\n'.format(r,c,v))
    f.close()
    #build graph
    g = Graph()
    g.read_edgelist(filename='tmp.txt', weighted=True, directed=True)
    #embedding
    model = node2vec.Node2vec(graph=g, path_length=10,num_paths=5, dim=10,workers=4, window=5, dw=True)
    # return embedding
    print(type(model.vectors)) 
    return model

path_ = '../../data/processed/walking_embedding'
#for t in range(1383260400000,1383270400000,600000*3):
for t in range(1383260400000,1385851800001,600000*3):
    em = deepwalk_embedding(t)
    em.save_embeddings('{}/{}.txt'.format(path_,t))
    if t % 10000000 == 0:
        print('interval {} finished...'.format(t))
