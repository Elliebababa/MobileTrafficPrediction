from os import *
import pandas as pd
import numpy as np

walking_embedding_path = '../../data/processed/walking_embedding'
def fetch_spatio_Embedding(time, grid,dim = 10):
    fp_ =  '{}/{}.txt'.format(walking_embedding_path,time)
    if os.path.exists(fp_):
        d = pd.read_table(fp_,sep= ' ',skiprows = 1, header = None, index_col = 0)
        if grid in d.index:
             #print('succeed in fetching embedding of grid {} at the time interval{}'.format(grid, time))
            return d.loc[grid].values
    return np.zeros(dim)

grid = {} 

dir_ = '../../data/processed/walking_embedding'
for _, _, file_ in walk(dir_):
    file_.sort()
    for i,f in enumerate(file_):
        fp_ = '{}/{}'.format(dir_,f)
        t = f[:-4]
        d = pd.read_table(fp_, sep = ' ', skiprows = 1, header = None, index_col = 0)
        for grid_idx in range(1,10001):
            if not grid_idx in grid:
                grid[grid_idx] = pd.DataFrame()
            if grid_idx in d.index:
                grid[grid_idx][t] = d.loc[grid_idx]
            else:
                grid[grid_idx][t] = 0
        print('finish {} files , time {}'.format(i,t))
    
fp_ = '../../data/processed/grid_embedding'    
for grid,df in grid.items():
    df.T.to_csv('{}/{}.csv'.format(fp_,grid),header = None)
        


