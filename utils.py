'''
This module contains general useful functions for the AeroCT package

Created on Jul 19, 2018

@author: savis
'''
from __future__ import division
import os
from datetime import datetime, timedelta
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from aeroct import aeronet
from aeroct import modis
from aeroct import metum

SCRATCH_PATH = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/'


def datetime_list(initial_date, final_date=None, days=None, str_format=None):
    '''
    Return a list of datetimes (at 00:00:00) or strings beginning at 'initial_date'.
    If 'final_date' is given then all the days up to this day will be returned. Otherwise
    the days argument is used. If both are None then all dates up to the yesterday are
    returned.
    
    Parameters
    ----------
    initial_date : str
        The first date in the list. Format is 'YYYYMMDD'.
    initial_date : str, optional (Default: None)
        The final date in the list. Format is 'YYYYMMDD'.
    days : list of int, optional (Default: None)
        The days after initial_date to return datetime objects.
        Eg. datetime_list(initial_date='20180624', days=[-3,4,7]) will return datetime a
        list of datetime objects of: [2018-06-21, 2018-06-28, 2018-07-02]
    str_format : str, optional (Default: None)
        If None then datetimes will be returned otherwise this string provides the format
        to convert them to datetimes.
    '''
    if final_date is not None:
        days = (datetime.strptime(final_date, '%Y%m%d') - \
                datetime.strptime(initial_date, '%Y%m%d')).days
        days = range(days + 1)
    elif days is None:
        days = (datetime.utcnow() - datetime.strptime(initial_date, '%Y%m%d')).days
        days = range(days)
    
    initial_date = datetime.strptime(initial_date, '%Y%m%d')
    dt_list = [initial_date + timedelta(days=int(d)) for d in days]
    
    if str_format != None:
        dt_list = [dt.strftime(str_format) for dt in dt_list]
    return dt_list


def download_range(data_set, date_list, dl_dir=SCRATCH_PATH+'downloads/',
                   forecast_time=0, src=None, dl_again=False):
    '''
    Download the files containing data for a list of dates. Then the required data fields
    are extracted and the data is saved in a pickled file for each date.
    
    Parameters
    ----------
    data_set : str
        The data set to load. This may be 'aeronet', 'modis', 'modis_a', 'modis_t',
        or 'metum'.
    date_list : datetime list
        The dates for the data that is to be downloaded.
    dl_dir : str, optional (Default: '/scratch/{USER}/aeroct/downloads/')
        The directory in which to save downloaded data. The different data sets will be
        saved within directories in this location.
    forecast_time: int, optional (Default: 0)
        The forecast lead time to use if 'metum' is chosen.
    src : str, optional (Default: None)
        The source to retrieve the data from.
        MODIS: 'NASA' or None to use ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/
               'MetDB' for MetDB extraction (Note: fewer dust filters available)
    dl_again : bool, optional (Default: False)
        If it is True then it will download the data again, even if the file already
        exists.
    '''
    
    # Get correct sub-directory to download into
    if dl_dir[-1] != '/': dl_dir += '/'
    
    if data_set == 'aeronet':        
        dl_dir = dl_dir + 'AERONET/'
    elif data_set == 'metum':        
        dl_dir = dl_dir + 'UM/'
    elif data_set[:5] == 'modis': 
        dl_dir = dl_dir + 'MODIS/'
    else:
        raise ValueError('Invalid data set: {0}'.format(data_set))
    
    if data_set == 'modis':
        satellite = 'Both'
    elif data_set == 'modis_t':
        satellite = 'Terra'
    elif data_set == 'modis_a':
        satellite = 'Aqua'
    
    ds_names = {'aeronet': 'AERONET',
                'modis': 'MODIS',
                'modis_t' : 'MODIS_Terra',
                'modis_a' : 'MODIS_Aqua',
                'metum': 'Unified_Model'}
    ds_name = ds_names[data_set]
    
    # Download the files at once for the UM
    if data_set == 'metum':
        metum.download_data_range(date_list, forecast_time, dl_dir + 'pp/', dl_again)    
    
    for date in date_list:
        
        # Filepaths to save data under
        if data_set == 'aeronet':
            filepath = '{0}AERONET_{1}'.format(dl_dir, date)
            no_file = (not os.path.exists(filepath))
        elif data_set == 'metum':
            filepath = '{0}Unified_Model{1:03d}_{2}'.format(dl_dir, forecast_time, date)
            no_file = (not os.path.exists(filepath))
        elif data_set[:5] == 'modis':
            filepath = '{0}{1}_{2}'.format(dl_dir, ds_name, date)
            modis_filepath = '{0}MODIS_{1}'.format(dl_dir, date)
            no_file = (not os.path.exists(filepath)) & (not os.path.exists(modis_filepath))
            
        if no_file | dl_again:
            
            # Download and load data
            if data_set == 'aeronet':
                print('Downloading AERONET data for {0}.'.format(date))
                dl_data = aeronet.download_data_day(date)
            
            elif data_set == 'metum':
                print('Loading Unified Model files.')
                dl_data = metum.load_files(date, forecast_time, dl_dir + 'pp/')
            
            elif data_set[:5] == 'modis':
                if (src == 'MetDB'):
                    print('Extracting {0} data from MetDB for {1}.'.format(ds_name, date))
                    dl_data = modis.retrieve_data_day_metdb(date, satellite)
                
                elif (src is None) | (src == 'NASA'):
                    print('Loading {0} data for {1}.'.format(ds_name, date))
                    dl_data = modis.load_data_day(date, dl_dir=dl_dir+'hdf/',
                                                satellite=satellite, dl_again=dl_again)
                
            # Save data
            if not os.path.exists(dl_dir):
                os.makedirs(dl_dir)
            
            os.system('touch {0}'.format(filepath))
            with open(filepath, 'w') as w:
                print('Saving data to {0}'.format(filepath))
                pickle.dump(dl_data, w, -1)


def flattend_3D_grid(x, y, z):
    '''
    Produce a grid for 3 axes, similarly to np.meshgrid. This is then flattened
    '''
    len1, len2, len3 = len(x), len(y), len(z)
    x = x.reshape(len1, 1, 1)
    y = y.reshape(1, len2, 1)
    z = z.reshape(1, 1, len3)
    
    X = x.repeat(len2, axis=1).repeat(len3, axis=2)
    Y = y.repeat(len1, axis=0).repeat(len3, axis=2)
    Z = z.repeat(len1, axis=0).repeat(len2, axis=1)
    
    return X.flatten(), Y.flatten(), Z.flatten()