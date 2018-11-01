import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

def DTWdistance(s1, s2, w = 48):
    w = max(w, abs(len(s1)-len(s2)))
    DTW = {}
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i,j)] = float('inf')
    DTW[(-1,-1)] = 0
    
    for i in range(len(s1)):
        for j in range(max(0,i-w),min(len(s2),i+w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i,j)] = dist + min(DTW[(i-1,j)],DTW[(i, j-1)], DTW[i-1,j-1])
    
    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def loadgrids(gridNum = 10000, path_to_file = '../../data/processed/gridTraffic30/gridTraffic30'):
    print('loading '+ str(gridNum) +' grids from ' + path_to_file)
    grids = {}
    for i in range(1,gridNum+1):
        tmp = pd.read_csv('{}/grid{}.csv'.format(path_to_file,i))
        grids[i-1] = tmp.traffic
    print('finish loading grids from ' + path_to_file)
    return grids

def gen_dtw_matrix(grids, gridNum = 10000):
    print('generating matrix...')
    str_time = time.time()
    mat = np.zeros((gridNum,gridNum))
    for i in range(gridNum):
        print('grid'+str(i)+'....')
        for j in range(i+1,gridNum):
            mat[i][j] = mat[j][i] = DTWdistance(grids[i],grids[j])
        print('{} seconds been spent..'.format(time.time()-str_time))
    print('done...')
    return mat

if __name__ == "__main__":
    gridNum = 5
    grids = loadgrids(gridNum)
    mat = gen_dtw_matrix(grids,gridNum)
    np.save('dtw_matrix_{}.npy'.format(gridNum), mat)



'''
NO WINDOW VERSION:

generating matrix...
grid0....
190.7216715812683 seconds been spent..
grid1....
334.9178419113159 seconds been spent..
grid2....
430.1747190952301 seconds been spent..
grid3....
474.3861689567566 seconds been spent..
grid4....
474.38625836372375 seconds been spent..
done...

W = 48
grid0....
16.147517442703247 seconds been spent..
grid1....
28.725412607192993 seconds been spent..
grid2....
37.04055070877075 seconds been spent..
grid3....
41.171730756759644 seconds been spent..
grid4....
41.17181420326233 seconds been spent..
done...


'''
