'''
Created on Jun 22, 2018

@author: savis

TODO: Add an attribute for if the data frames require testing for nearby times or
    locations.
'''

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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
                 forecast_time=None, data_set=None, grid=False):
        self.data = data                    # AOD data
        self.longitudes = longitudes        # [degrees]
        self.latitudes = latitudes          # [degrees]
        self.times = times                  # [hours since 00:00:00 on date]
        
        self.date = date                    # (datetime)
        self.wavelength = wavelength        # [nm]
        self.forecast_time = forecast_time  # [hours]
        self.grid = grid                    # Has a grid in space and time? (forecast)
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
            Default: '{data_set}_YYYYMMDD_##'
        '''
        # Make directory if it does not exist
        os.system('mkdir -p {}'.format(path))
        
        if filename != None:
            pass
        elif type(self.data_set) == str:
            filename = '{}_{}_'.format(self.data_set, self.date.strftime('%Y%m%d'))
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
        
    def plot_map(self, grid_size=0.01):
        if self.grid == False:
            lat_grid_bounds = np.arange(- 90,  90 + grid_size/2, grid_size)
            lon_grid_bounds = np.arange(-180, 180 + grid_size/2, grid_size)
            
            
            ax = plt.axes(projection=ccrs.PlateCarree())
            plt.scatter(self.latitudes, self.longitudes, c=self.data)
            ax.coastlines()
            plt.show()


class MatchFrame():
    '''
    The data frame into which the matched AOD data is processed for a single day.
    The averaged AOD data and standard deviations are contained within 2D numpy array,
    with the first index referring to the data set. The latitudes, longitudes, and times
    are all stored in 1D numpy arrays, all of the same length. The date, wavelength and
    forecast time (for models) are stored as attributes.
    '''

    def __init__(self, data, data_std, data_num, latitudes, longitudes, times, date,
                 wavelength=550, forecast_times=(None, None), data_sets=(None, None),
                 grid=False):
        self.data = data                    # Averaged AOD data
        self.data_std = data_std            # Averaged AOD data standard deviations
        self.data_num = data_num            # Number of values that are averaged
        self.longitudes = longitudes        # [degrees]
        self.latitudes = latitudes          # [degrees]
        self.times = times                  # [hours since 00:00:00 on date]
        
        self.date = date                    # (datetime)
        self.wavelength = wavelength        # [nm]
        self.forecast_times = forecast_times# [hours] tuple
        self.grid = grid                    # Has a grid in space and time? (forecast)
        self.data_sets = data_sets          # A tuple of the names of the data sets
    
    def datetimes(self):
        return [self.date + timedelta(hours=h) for h in self.times]
    
    
    def dump(self, path=scratch_path, filename=None):
        '''
        Save the data frame as a file in the chosen location. Note that saving and
        loading large data frames can take some time. The filename is returned.
        
        Parameters:
        path: (str, optional) The path to the directory where the file will be saved.
            Default: '/scratch/{USER}/aeroct/data_frames/'
        filename: (str, optional) What to name the saved file.
            Default: '{data_sets}_YYYYMMDD_##'
        '''
        # Make directory if it does not exist
        os.system('mkdir -p {}'.format(path))
        
        if filename == None:
            if type(self.data_sets) == tuple:
                filename = '{}-{}_{}_'.format(self.data_sets[0], self.data_sets[1],
                                                    self.date.strftime('%Y%m%d'))
            else:
                raise ValueError, 'data_sets attribute invalid. Cannot create filename'
        
        i = 0
        while os.path.exists(path + filename + str(i).zfill(2)):
            i += 1
        
        # Write file
        os.system('touch {}'.format(path + filename + str(i).zfill(2)))
        with open(path + filename + str(i).zfill(2), 'w') as writer:
            pickle.dump(self, writer, -1)
        print('Data frame saved successfully to {}'.format(path + filename + str(i).zfill(2)))
        
        return filename
    
    
    # Stats
    rms = lambda self: np.average(self.data[1] - self.data[0])
    
    
    def scatter_plot(self, stats=True, show=True, error=True):
        '''
        This is used to plot AOD data from two sources which have been matched-up on a
        scatter plot. The output are the stats if stats=True and the figure also if
        show=True.
        
        Parameters:
        stats: (bool, optional) Choose whether to calculate statistics for the plot.
            These are shown on the plot if show=True.    Default: True
        show: (bool, optional) If True, the plot is shown otherwise the figure is passed
            as an output.    Default: True
        error: (bool, optional) If True, error bars for the standard deviations are
            included on the plot.
        '''
        if len(self.data_sets) != 2:
            raise ValueError, 'The data frame must be matched-up data from two data sets'
        
        fig = plt.figure()
        
        if error == True:
            plt.errorbar(self.data[0], self.data[1], self.data_std[1], self.data_std[0], 'r.', ecolor='gray')
        else:
            plt.plot(self.data[0], self.data[1], 'r.')
        
        high = np.nanmax(self.data)
        low = np.nanmin(self.data)
        diff = high - low
        plt.plot([low, high], [low, high], 'k--')
        
        plt.title(self.date)
        plt.xlabel(self.data_sets[0])
        plt.ylabel(self.data_sets[1])
        plt.loglog()
        plt.xlim(low, high)
        plt.ylim(low, high)
        
        rms = self.rms()
        plt.text(low + 0.01*diff, high - 0.1*diff, 'RMS: {:.02f}'.format(rms), fontsize=15)
        
        if show == True:
            plt.show()
            return [rms]
        else:
            return fig, [rms]


def load(data_set, date, forecast_time=0, src=None, out_dir=scratch_path, save=True):
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
        if not os.path.exists('{}AERONET_{}'.format(out_dir, date)):
            print('Downloading AERONET data for ' + date +'.')
            aod_string = aeronet.download_data_day(date)
            if save == True:
                with open('{}AERONET_{}'.format(out_dir, date), 'w') as w:
                    pickle.dump(aod_string, w, -1)
        else:
            with open('{}AERONET_{}'.format(out_dir, date), 'r') as r:
                    aod_string = pickle.load(r)
        
        aod_df = aeronet.parse_data(aod_string)
        print('Processing...', end='')
        parameters = aeronet.process_data(aod_df, date)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set)
    
    elif data_set == 'modis':
        if not os.path.exists('{}MODIS_{}'.format(out_dir, date)):
            print('Downloading MODIS data for ' + date +'.')
            aod_array = modis.retrieve_data_day(date)
            if save == True:
                with open('{}MODIS_{}'.format(out_dir, date), 'w') as w:
                    pickle.dump(aod_array, w, -1)
        else:
            with open('{}MODIS_{}'.format(out_dir, date), 'r') as r:
                    aod_array = pickle.load(r)
        
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


def load_from_file(filename, path=scratch_path):
        '''
        Load the data frame from a file in the chosen location. Note that saving and
        loading large data frames can take some time.
        
        Parameters:
        filename: (str) The name of the saved file.
        path: (str, optional) The path to the directory from which to load the file.
            Default: '/scratch/{USER}/aeroct/data_frames/'
        '''
        if not os.path.exists(path + filename):
            raise ValueError, 'File does not exist: {}'.format(path + filename)
        
        with open(path + filename, 'r') as reader:
            data_frame = pickle.load(reader)
        return data_frame