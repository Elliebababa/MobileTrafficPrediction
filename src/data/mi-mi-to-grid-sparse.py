#a module used to transform the mi-mi to grid data
#the data is grouped by day. i.e. output daily interaction between grids
import pandas as pd
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
import click
import logging
from pathlib import Path
from os import walk

@click.command()
@click.argument('input_filepath', type = click.Path(exists = True))
@click.argument('output_filepath',type = click.Path())

def main(input_filepath = None, output_filepath = None):
    #caltheInteractionPerday()
    
    caltheInteractionPer10min(input_filepath, output_filepath)

def caltheInteractionPerday(input_filepath = None, output_filepath = None):
    logger = logging.getLogger(__name__)
    logger.info('transforming the raw mi-mi data to grid data..then we will get the call interactions among grids per day...')

    names = ['time','from','to','strength']
    dir_ = input_filepath

    for _,_, file in walk(dir_):
        for fp in file:
            S = dok_matrix((10001,10001),dtype = np.float32)
            with open(dir_+'/'+fp,'r') as f:
                for line_terminated in f:
                    line = line_terminated.rstrip('\n').split()
                    from_,to_,data = int(line[1]),int(line[2]),float(line[3])
                    if (from_,to_) in S:
                        S[from_,to_] += data
                    else:
                        S[from_,to_] = data
            Scsr = S.tocsr()
            scipy.sparse.save_npz(output_filepath+'/'+fp.replace("txt", "npz"),S.tocsr())
            logging.info(fp+' processing done..')
            '''
            data = pd.read_table(dir_+'/'+f, names = names)
            print(f)
            row = np.array([])
            col = np.array([])
            stre = np.array([])
            for (from_,to_),d in data.groupby(['from','to']):
                row = np.append(row,from_)
                col = np.append(col,to_)
                stre = np.append(stre,d.strength.sum())
            cm = csr_matrix((stre,(row,col)),shape = (10001,10001))
            scipy.sparse.save_npz(output_filepath+'/'+f.replace("txt", "npz"),cm)
            logging.info(f+' processing done..')
            
            datagroup = data.groupby(['from','to']).sum()
            datagroup = datagroup.drop(['time'],axis = 1)
            datagrid = datagroup.unstack().fillna(0)
            datagrid.to_csv(output_filepath+'/'+f.replace("txt", "csv"),header = None, index = None)
            logging.info(f+' processing done..')
            '''
def caltheInteractionPer10min(input_filepath = None, output_filepath = None):
    logger = logging.getLogger(__name__)
    logger.info('transforming the raw mi-mi data to grid data..then we will get the call interactions among grids per 10 min.')

    #names = ['time','from','to','strength']
    dir_ = input_filepath

    for _,_, file in walk(dir_):
        for fp in file:
            with open(dir_+'/'+fp,'r') as f:
                timeIntervalsheets = {}
                for line_terminated in f:
                    line = line_terminated.rstrip('\n').split()
                    t, from_,to_,data = int(line[0]), int(line[1]),int(line[2]),float(line[3])
                    if t in timeIntervalsheets:
                        if (from_, to_) in timeIntervalsheets[t]:
                            timeIntervalsheets[t][from_,to_] +=data
                        else:
                            timeIntervalsheets[t][from_,to_] = data
                    else:
                        timeIntervalsheets[t] = dok_matrix((10001,10001),dtype = np.float32)
                for t in timeIntervalsheets:
                    S = timeIntervalsheets[t].tocsr()
                    scipy.sparse.save_npz('{}/{}.npz'.format(output_filepath,t),S)
                    logging.info('Interval'+ str(t)+' processing done..')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()