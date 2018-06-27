'''
Created on Jun 22, 2018

@author: savis

TODO: Add an attribute for if the data frames require testing for nearby times or
    locations.
'''

from __future__ import print_function, division
import numpy as np
from datetime import datetime, timedelta

import aeronet
import modis
import metum

class DataFrame():
    '''
    The data frame into which the AOD data is processed. Only a single day's data and
    some from the days before and after are included.
    The AOD data, latitudes, longitudes, and times are all stored in 1D numpy arrays, all
    of the same length. The date, wavelength and forecast time (for models) are stored
    as attributes. So each forecast time and date requires a new instance.
    '''

    def __init__(self, data, latitudes, longitudes, times, date, wavelength=550,
                 forecast_time=None, data_set=None):
        self.data = data                    # AOD data
        self.longitudes = longitudes        # [degrees]
        self.latitudes = latitudes          # [degrees]
        self.times = times                  # [hours since 00:00:00 on date]
        
        self.date = date                    # (datetime)
        self.wavelength = wavelength        # [nm]
        self.forecast_time = forecast_time  # [hours]
    
    def datetimes(self):
        return [self.date + timedelta(hours=h) for h in self.times]


def load_data_frame(data_set, date, forecast_time=0, src=None, out_dir=None):
    '''
    Load a data frame for a given date using data from either AERONET, MODIS, or the
    Unified Model (metum). This will allow it to be matched and compared with other data
    sets.
    
    Parameters:
    data_set: (str) The data set to load. This may be 'aeronet', 'modis', or 'metum'.
    date: (str) The date for the data that is to be loaded. Specify in format 'YYYYMMDD'.
    forecast_time: (int, optional) The number of hours ahead for the forecast if metum is
        chosen. (Default: 0)
    src: (str, optional) The source to retrieve the data from.
        (Currently unavailable)
    out_dir: (str, optional) The directory in which to save any files.
        (Currently unavailable)
    '''
    
    if data_set == 'aeronet':
        print('Downloading AERONET data for ' + date +'.')
        aod_string = aeronet.download_data_day(date)
        aod_df = aeronet.parse_data(aod_string)
        print('Processing...', end='')
        parameters = aeronet.process_data(aod_df, date)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set)
    
    elif data_set == 'modis':
        print('Downloading MODIS data for ' + date +'.')
        aod_array = modis.retrieve_data_day(date)
        print('Processing...', end='')
        parameters = modis.process_data(aod_array, date)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set)
    
    elif data_set == 'metum':
        print('Retrieving Unified Model files for ' + date + '.')
        metum.download_data_day(date)
        print('Loading files.')
        aod_cube = metum.load_files(date, forecast_time, out_dir)
        print('Processing...', end='')
        parameters = metum.process_data(aod_cube, forecast_time)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set)
    
    else:
        print('Invalid data set: {}'.format(data_set))
        return