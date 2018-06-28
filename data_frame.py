'''
Created on Jun 22, 2018

@author: savis

TODO: Add an attribute for if the data frames require testing for nearby times or
    locations.
'''

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import aeronet
import modis
import metum

scratch_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/data_frames/'

class DataFrame():
    '''
    The data frame into which the AOD data is processed. Only a single day's data and
    some from the days before and after are included.
    The AOD data, latitudes, longitudes, and times are all stored in 1D numpy arrays, all
    of the same length. The date, wavelength and forecast time (for models) are stored
    as attributes. So each forecast time and date requires a new instance.
    '''

    def __init__(self, data, latitudes, longitudes, times, date, wavelength=550,
                 forecast_time=None, data_set=None, gridded=False):
        self.data = data                    # AOD data
        self.longitudes = longitudes        # [degrees]
        self.latitudes = latitudes          # [degrees]
        self.times = times                  # [hours since 00:00:00 on date]
        
        self.date = date                    # (datetime)
        self.wavelength = wavelength        # [nm]
        self.forecast_time = forecast_time  # [hours]
        self.gridded = gridded              # Has a grid in space and time? (forecast)
        self.data_set = data_set            # The name of the data set or a tuple
    
    def datetimes(self):
        return [self.date + timedelta(hours=h) for h in self.times]
    
    
    def dump(self, path=scratch_path, filename=None):
        '''
        Save the data frame as a file in the chosen location. Note that saving and
        loading large data frames can take some time.
        
        Parameters:
        path: (str, optional) The path to the directory where the file will be saved.
            Default: '/scratch/{USER}/aeroct/data_frames/'
        filename: (str, optional) What to name the saved file.
            Default: '{data_set(s)}_YYYYMMDD_##'
        '''
        # Make directory if it does not exist
        os.system('mkdir -p {}'.format(path))
        
        if filename != None:
            pass
        elif type(self.data_set) == str:
            filename = '{}_{}_'.format(self.data_set, self.date.strftime('%Y%m%d'))
        elif type(self.data_set) == tuple:
            filename = '{}-{}_{}_'.format(self.data_set[0], self.data_set[1],
                                                self.date.strftime('%Y%m%d'))
        else:
            raise ValueError, 'data_set attribute invalid. Cannot create filename'
        
        i = 0
        while os.path.exists(path + filename + str(i).zfill(2)):
            i += 1
        
        # Write file
        os.system('touch {}'.format(path + filename + str(i).zfill(2)))
        with open(path + filename + str(i).zfill(2), 'w') as writer:
            pickle.dump(self, writer, -1)
        print('Data frame saved successfully to {}'.format(path + filename + str(i).zfill(2)))
    
    
    def scatter_plot(self, stats=True, show=True):
        '''
        This is used to plot AOD data from two sources which have been matched-up on a
        scatter plot. The output are the stats if stats=True and the figure also if
        show=True.
        
        Parameters:
        stats: (bool, optional) Choose whether to calculate statistics for the plot.
            These are shown on the plot if show=True.    Default: True
        show: (bool, optional) If True, the plot is shown otherwise the figure is passed
            as an output.    Default: True
        '''
        if type(self.data_set) != tuple:
            raise ValueError, 'The data frame must be matched-up data from two data sets'
        
        fig = plt.figure()
        plt.plot(self.data[0], self.data[1], '.', color=self.times/24)
        max_value = np.max(self.data)
        plt.plot([0, max_value], [0, max_value], 'b--')
        
        if show is True:
            fig.show()
            return
        else:
            return fig


def load(data_set, date, forecast_time=0, src=None, out_dir=None):
    '''
    Load a data frame for a given date using data from either AERONET, MODIS, or the
    Unified Model (metum). This will allow it to be matched and compared with other data
    sets.
    
    Parameters:
    data_set: (str) The data set to load. This may be 'aeronet', 'modis', or 'metum'.
    date: (str) The date for the data that is to be loaded. Specify in format 'YYYYMMDD'.
    forecast_time: (int, optional) The forecast leading time to use if metum is chosen.
        (Default: 0)
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
        metum.download_data_day(date, forecast_time)
        print('Loading files.')
        aod_cube = metum.load_files(date, forecast_time, out_dir)
        print('Processing...', end='')
        parameters = metum.process_data(aod_cube, date, forecast_time)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set, gridded=True)
    
    else:
        print('Invalid data set: {}'.format(data_set))
        return
