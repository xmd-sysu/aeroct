'''
This module contains general useful functions for the AeroCT package

Created on Jul 19, 2018

@author: savis
'''
from __future__ import division, print_function
import os
from datetime import datetime, timedelta
import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import aeroct
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


def get_match_list(data_set1, data_set2, date_list, save=True, save_dir=SCRATCH_PATH+'match_frames/',
                   **kwargs):
    '''
    This will return a list of MatchFrames for matched data-sets for a corresponding
    list of dates. The data will be initially downloaded if it has not been already and each
    individual MatchFrame will be saved unless stated otherwise. Any existing saved
    MatchFrames will be loaded automatically.
    
    Parameters:
    -----------
    data_set1, data_set2 : {'metum', 'modis', 'modis_a', 'modis_t', 'aeronet'}
        The data-sets that should be matched.
    date_list : list of datetimes
        This provides the dates over which data should be downloaded and matched. The
        aeroct.datetime_list() function can be used to provide these.
    save : bool, optional (Default: True)
        If True then each MatchFrame will be saved as a pickled object using the dump()
        method.
    save_dir : str, optional (Default: '/scratch/{USER}/aeroct/match_frames/')
        The directory within which to save the match_frames.
    kwargs:
    forecast_time1 = int (Default: True)
        The forecast time if the chosen data-set1 is a model. (Only multiples of 3)
    forecast_time2 = int (Default: True)
        The forecast time if the chosen data-set2 is a model. (Only multiples of 3)
    match_time = int (Default: 30)
        The difference in time over which to match data (hours).
    match_rad = int (Default: 25)
        The difference in space over which to match data (km).
    aod_type : {'total', 'dust'} (Default: 'total')
        The type of AOD data to match-up if there is a choice (MODIS).
    dl_again : bool (Default: False)
        If the data should be downloaded and matched again even if files for it exist.
    dl_dir : str (Default: '/scratch/{USER}/aeroct/downloads/')
        The directory within which to save downloaded data.
    subdir : bool (Default: True)
        If different datasets should be saved within subdirectories within 'save_dir'.
    '''
    
    if save_dir[-1] != '/': save_dir += '/'
    
    # kwargs
    fc_time1 = kwargs.setdefault('forecast_time1', None)
    fc_time2 = kwargs.setdefault('forecast_time2', None)
    match_time = kwargs.setdefault('match_time', 30)
    match_rad = kwargs.setdefault('match_rad', 25)
    aod_type = kwargs.setdefault('aod_type', 'total')
    dl_again = kwargs.setdefault('dl_again', False)
    dl_dir = kwargs.setdefault('dl_dir', SCRATCH_PATH+'downloads/')
    subdir = kwargs.setdefault('subdir', True)
    
    if (data_set1 == 'metum') & (fc_time1 is None):
        raise ValueError('Data set 1 requires a forecast time, none given.')
    if (data_set2 == 'metum') & (fc_time2 is None):
        raise ValueError('Data set 2 requires a forecast time, none given.')
    
    # Put the forecast time onto the end of model data-set names for the filename
    data_set_names = []
    fc_times = [fc_time1, fc_time2]
    for i, data_set in enumerate([data_set1, data_set2]):
        if fc_times[i] is not None:
            fc_str = str(int(fc_times[i])).zfill(3)
        else:
            fc_str = ''
        data_set_names.append(data_set + fc_str)
    
    # AOD type letter for filenames
    if ('metum' in [data_set1, data_set2]) | (aod_type == 'dust'):
        aod_s = 'd'
    else:
        aod_s = 't'
    
    # Subdirectories
    if subdir:
        subdir_path = '{0}-{1}-{2}/'.format(data_set_names[1], data_set_names[0], aod_s)
    else:
        subdir_path = ''
    
    # Build the list of MatchFrames
    mf_list = []
    for date in date_list:
        print('\nDate: {0}'.format(date.date()))
        
        # Load pickled MatchFrame if it already exists
        filename0 = '{0}-{1}-{2}-{3}.pkl'.format(data_set_names[1], data_set_names[0],
                                                 aod_s, date.strftime('%Y%m%d'))
        if os.path.exists(save_dir + subdir_path + 'pkl/' + filename0) & (not dl_again):
            mf_list.append(aeroct.load_from_pickle(filename0,
                                                   save_dir+subdir_path+'pkl/'))
        
        else:
            df1 = aeroct.load(data_set1, date, forecast_time=fc_time1, dl_dir=dl_dir,
                              dl_again=dl_again, verb=False)
            
            df2 = aeroct.load(data_set2, date, forecast_time=fc_time2, dl_dir=dl_dir,
                              dl_again=dl_again, verb=False)
            
            mf = aeroct.collocate(df1, df2, match_time, match_rad, aod_type=aod_type,
                                  save=save, save_dir=save_dir, save_subdir=subdir)
            mf_list.append(mf)
    
    return mf_list


def average_each_n_values(array, n):
    array = np.array(array)
    padded_array = np.pad(array.astype(float), (0, (n - array.size%n)%n),
                          mode='constant', constant_values=np.NaN)
    averaged_array = np.nanmean(padded_array.reshape(-1, n), axis=1)
    averaged_array_std = np.nanstd(padded_array.reshape(-1, n), axis=1)
    return averaged_array, averaged_array_std