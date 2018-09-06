# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
from os import walk
import numpy as np
#from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath = None, output_filepath = None):
    """ Runs data processing scripts to turn raw data from (../../data/raw) into
        cleaned data ready to be analyzed (saved in ../../data/processed).
        usage example: python make_dataset.py ../../data/raw ../../data/processed
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    Nov = loadRaw(input_filepath+'/milan/sms-call-internet-mi/sms-call-internet-mi-Nov')
    aggregateGridData(Nov,output_filepath+'/gridTraffic')

def loadRaw(filepath = None):
    sheetList = []
    names = ['squareId', 'timeInterval', 'countryCode', 'smsIn', 'smsOut', 'callIn', 'callOut', 'Internet']
    dir_ = filepath
    for _, _, file in walk(dir_):
        for f in file:
            data = pd.read_table(dir_ + '/' + f, names=names)
            sheetList.append(data)
        print('Reading Finished. There are ' + str(len(sheetList)) + ' files in all.')
    return sheetList

def aggregateGridData(sheetList,  output_filepath ,header = None, index = None):
    # used to aggregate the data of each grid
    # the output file named 'grid#.csv', no headers and no index by default
    # csv format timeInterval,callin,callout,smsin,smsout,internet
    logger = logging.getLogger(__name__)
    gridNum = 10000
    for i in range(1, gridNum+1):
        gridData = pd.DataFrame()
        for sheet in sheetList:
            tmp = sheet[sheet['squareId'] == i]
            gridData = pd.concat([gridData, tmp], axis=0).reset_index(drop=True)

        #aggregate data
        gd = gridData.groupby(['timeInterval']).sum()
        gd = gd.drop(['countryCode','squareId'],axis = 1)#.reset_index(drop = True)

        #deal with missing data
        gd = gd.fillna(0)


        '''not all the time are recorded in the dataset, so we need to check and insert those missing interval'''

        #check and insert missing timeInterval
        tt = gd.index
        for id, time in enumerate(tt[:-1]):
            if not tt[id + 1] == time + 600000:
                for missingInterval in range(time + 600000, tt[id + 1],600000):
                    logging.warning('warning: missing time interval '+str(missingInterval)+'... and now inserting...')
                    gd.loc[missingInterval] = [0,0,0,0,0]

        gd = gd.sort_index()
        #write csv to file
        gd.to_csv(output_filepath +'/grid'+str(i)+'.csv',header = header)
        logging.info('grid'+str(i)+' aggregated\n')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
