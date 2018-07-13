'''
Created on Jun 22, 2018

@author: savis

'''
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from scipy import stats
import os
import warnings
from scipy.interpolate import griddata
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import aeronet
import modis
import metum

scratch_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/'

# How to output the names of the data sets
name = {'aeronet': 'AERONET',
        'modis': 'MODIS',
        'modis_t' : 'MODIS_Terra',
        'modis_a' : 'MODIS_Aqua',
        'metum': 'Unified_Model'}


class DataFrame():
    '''
    The data frame into which the AOD data is processed. Only a single day's data and
    some from the days before and after are included.
    The AOD data, latitudes, longitudes, and times are all stored in 1D numpy arrays, all
    of the same length. The date, wavelength and forecast time (for models) are stored
    as attributes. So each forecast time and date requires a new instance.
    '''

    def __init__(self, aod, aod_d, latitudes, longitudes, times, date, wavelength=550,
                 data_set=None, **kw):
        # Ensure longitudes are in range [-180, 180]
        longitudes = longitudes.copy()
        longitudes[longitudes > 180] -= 360
        
        self.aod        = aod                       # Total AOD data
        self.aod_d      = aod_d                     # Dust component of AOD data
        self.longitudes = longitudes                # [degrees]
        self.latitudes  = latitudes                 # [degrees]
        self.times      = times                     # [hours since 00:00:00 on date]
        
        self.date       = date                      # (datetime)
        self.wavelength = wavelength                # [nm]
        self.data_set   = data_set                  # The name of the data set
        self.forecast_time = kw.setdefault('forecast_time', None)  # [hours]
        self.cube = kw.setdefault('cube', None)    # contains iris cube if from_cube() used
    
    
    @classmethod
    def from_cube(cls, cube, data_set):
        # Create a DataFrame using a cube containing model data (dust AOD only)
        aod_d = cube.data
        lons = cube.coord('longitude').points
        lats = cube.coord('latitude').points
        times = cube.coord('time').points
        
        date = cube.coord('date').points[0]
        wl = cube.coord('wavelength').points[0]
        fc_time = cube.coord('forecast_time').points[0]
        
        return cls(None, aod_d, lats, lons, times, date, wl, data_set,
                   forecast_time=fc_time, cube=cube)
    
    
    def datetimes(self):
        return [self.date + timedelta(hours=h) for h in self.times]
    
    
    def dump(self, filename=None, dir_path=scratch_path+'data_frames/'):
        '''
        Save the data frame as a file in the chosen location. Note that saving and
        loading large data frames can take some time.
        
        Parameters:
        filename: str, optional (Default: '{data_set}_YYYYMMDD_##')
            What to name the saved file.
        dir_path: str, optional (Default: '/scratch/{USER}/aeroct/data_frames/')
            The path to the directory where the file will be saved.
            
        '''
        # Make directory if it does not exist
        os.system('mkdir -p {}'.format(dir_path))
        
        if filename != None:
            pass
        elif type(self.data_set) == str:
            filename = '{}_{}_'.format(self.data_set, self.date.strftime('%Y%m%d'))
        else:
            raise ValueError, 'data_set attribute invalid. Cannot create filename'
        
        i = 0
        while os.path.exists(dir_path + filename + str(i).zfill(2)):
            i += 1
        
        # Write file
        os.system('touch {}'.format(dir_path + filename + str(i).zfill(2)))
        with open(dir_path + filename + str(i).zfill(2), 'w') as writer:
            pickle.dump(self, writer, -1)
        print('Data frame saved successfully to {}'.format(dir_path + filename + str(i).zfill(2)))



class MatchFrame():
    '''
    The data frame into which the matched AOD data is processed for a single day.
    The averaged AOD data and standard deviations are contained within 2D numpy array,
    with the first index referring to the data set. The latitudes, longitudes, and times
    are all stored in 1D numpy arrays, all of the same length. The date, wavelength and
    forecast time (for models) are stored as attributes.
    '''

    def __init__(self, data, data_std, data_num, longitudes, latitudes, times, date,
                 match_time, match_rad, wavelength=550, forecast_times=(None, None),
                 data_sets=(None, None), aod_type=0, **kw):
        self.data       = data              # Averaged AOD data (Not flattend if cube != None)
        self.data_std   = data_std          # Averaged AOD data standard deviations
        self.data_num   = data_num          # Number of values that are averaged
        self.longitudes = longitudes        # [degrees]
        self.latitudes  = latitudes         # [degrees]
        self.times      = times             # [hours since 00:00:00 on date]
        
        self.date           = date            # (datetime)
        self.match_radius   = match_rad       # Maximum spacial difference between collocated points
        self.match_time     = match_time      # Maximum time difference between collocated points
        self.wavelength     = wavelength      # [nm]
        self.forecast_times = forecast_times  # [hours] tuple
        self.data_sets      = data_sets       # A tuple of the names of the data sets
        self.aod_type       = aod_type        # Whether it is coarse mode AOD or total AOD
        self.cube = kw.setdefault('cube', None)    # Contains AOD difference for a model-model match
        
        # Flattened
        self.data_f = np.array([self.data[0].ravel(), self.data[1].ravel()])
        self.std_f = np.array([self.data_std[0].ravel(), self.data_std[1].ravel()])
        
        # Stats
        self.RMS = np.sqrt(np.mean((self.data_f[1] - self.data_f[0])**2))   # Root mean square
        self.BIAS_MEAN = np.mean(self.data_f[1] - self.data_f[0])           # y - x mean
        self.BIAS_STD = np.std(self.data_f[1] - self.data_f[0])             # y - x standard deviation
        self.R_SLOPE, self.R_INTERCEPT, self.R = \
            stats.linregress(self.data_f[0], self.data_f[1])[:3]            # Regression and Pearson's correlation coefficient
    
    
    def datetimes(self):
        return [self.date + timedelta(hours=h) for h in self.times]
    
    
    def dump(self, filename=None, dir_path=scratch_path+'data_frames/'):
        '''
        Save the data frame as a file in the chosen location. Note that saving and
        loading large data frames can take some time. The filename is returned.
        
        Parameters:
        filename: str, optional (Default: '{data_sets}_YYYYMMDD_##')
            What to name the saved file.
        dir_path: str, optional (Default: '/scratch/{USER}/aeroct/data_frames/')
            The path to the directory where the file will be saved.
        '''
        # Make directory if it does not exist
        os.system('mkdir -p {}'.format(dir_path))
        
        if filename == None:
            
            if type(self.data_sets) == tuple:
                filename = '{}-{}_{}_'.format(self.data_sets[1], self.data_sets[0],
                                                  self.date.strftime('%Y%m%d'))
            else:
                raise ValueError, 'data_sets attribute invalid. Cannot create filename'
        
        i = 0
        while os.path.exists(dir_path + filename + str(i).zfill(2)):
            i += 1
        
        # Write file
        os.system('touch {}'.format(dir_path + filename + str(i).zfill(2)))
        with open(dir_path + filename + str(i).zfill(2), 'w') as writer:
            pickle.dump(self, writer, -1)
        print('Data frame saved successfully to {}'.format(dir_path + filename + str(i).zfill(2)))
        
        return filename



def load(data_set, date, forecast_time=0, src=None,
         dl_save=True, dl_dir=scratch_path+'downloads/'):
    '''
    Load a data frame for a given date using data from either AERONET, MODIS, or the
    Unified Model (metum). This will allow it to be matched and compared with other data
    sets.
    
    Parameters:
    data_set: str
        The data set to load. This may be 'aeronet', 'modis', or 'metum'.
    date: str
        The date for the data that is to be loaded. Specify in format 'YYYYMMDD'.
    forecast_time: int, optional (Default: 0)
        The forecast lead time to use if metum is chosen.
    src : str, optional (Default: None)
        The source to retrieve the data from.
        (Currently unavailable)
    dl_save : bool or str, optional (Default: True)
        Choose whether to save any downloaded data. If it is 'f' then it will download
        and save, even if the file already exists.
    dl_dir : str, optional (Default: '/scratch/{USER}/aeroct/downloads/')
        The directory in which to save downloaded data.
    '''
    if dl_dir[-1] != '/':
        dl_dir = dl_dir + '/'
    
    ds_name = name[data_set]
    
    if data_set == 'aeronet':
        dl_dir = dl_dir + 'AERONET/'
        filepath = '{0}{1}_{2}'.format(dl_dir, ds_name, date)
        
        if (not os.path.exists(filepath)) | (dl_save == 'f'):
            print('Downloading AERONET data for ' + date +'.')
            aod_string = aeronet.download_data_day(date)
            
            # Save data
            if (dl_save == True) | (dl_save == 'f'):
                
                if not os.path.exists(dl_dir):
                    os.makedirs(dl_dir)
                
                os.system('touch {}'.format(filepath))
                with open(filepath, 'w') as w:
                    print('Saving data to {}'.format(filepath))
                    pickle.dump(aod_string, w, -1)
        else:
            with open(filepath, 'r') as r:
                    aod_string = pickle.load(r)
        
        aod_df = aeronet.parse_data(aod_string)
        print('Processing AERONET data...', end='')
        parameters = aeronet.process_data(aod_df, date)
        print('Complete.\n')
        return DataFrame(*parameters, data_set=data_set)
    
    elif data_set[:5] == 'modis':
        if data_set == 'modis_t':
            satellite = 'Terra'
        elif data_set == 'modis_a':
            satellite = 'Aqua'
        else:
            satellite = 'Both'
        
        dl_dir_mod = dl_dir + 'MODIS/'
        filepath = '{0}{1}_{2}'.format(dl_dir_mod, ds_name, date)
        modis_filepath = '{}MODIS_{}'.format(dl_dir_mod, date)
        
        if (not os.path.exists(filepath)) & (not os.path.exists(modis_filepath)) | \
                                                                    (dl_save == 'f'):
            
            if (src == None) | (src == 'MetDB'):
                print('Extracting {} data from MetDB for {}.'.format(ds_name, date))
                aod_dict = modis.retrieve_data_day_metdb(date, satellite)
            
            elif src == 'NASA':
                print('Downloading {} data for {}.'.format(ds_name, date))
                aod_dict = modis.load_data_day(date,  dl_dir=dl_dir+'MODIS_hdf/',
                                               satellite=satellite, keep_files=True)
            
            # Save data
            if (dl_save == True) | (dl_save == 'f'):
                
                if not os.path.exists(dl_dir_mod):
                    os.makedirs(dl_dir_mod)
                
                os.system('touch {}'.format(filepath))
                with open(filepath, 'w') as w:
                    print('Saving data to {}.'.format(filepath))
                    pickle.dump(aod_dict, w, -1)
        elif os.path.exists(filepath):
            with open(filepath, 'r') as r:
                    aod_dict = pickle.load(r)
        else:
            with open(modis_filepath, 'r') as r:
                    aod_dict = pickle.load(r)
        
        print('Processing MODIS data...', end='')
        parameters = modis.process_data(aod_dict, date, satellite)
        print('Complete.\n')
        return DataFrame(*parameters, data_set=data_set)
    
    elif data_set == 'metum':
        dl_dir = dl_dir + 'UM/'
        
        force = False
        if dl_save == 'f':
            force = True
        
        metum.download_data_day(date, forecast_time, dl_dir, force)
        print('Loading Unified Model files.')
        aod_cube = metum.load_files(date, forecast_time, dl_dir)
        print('Processing Unified Model data...', end='')
        aod_cube = metum.process_data(aod_cube, date, forecast_time)
        print('Complete.\n')
        return DataFrame.from_cube(aod_cube, data_set)
    
    else:
        raise ValueError, 'Invalid data set: {}'.format(data_set)


def load_from_file(filename, dir_path=scratch_path+'data_frames'):
    '''
    Load the data frame from a file in the chosen location. Note that saving and
    loading large data frames can take some time.
    
    Parameters:
    filename: str
        The name of the saved file.
    dir_path: str, optional (Default: '/scratch/{USER}/aeroct/data_frames/')
        The path to the directory from which to load the file.
    '''
    if dir_path[-1] != '/':
        dir_path = dir_path + '/'
    
    if not os.path.exists(dir_path + filename):
        raise ValueError, 'File does not exist: {}'.format(dir_path + filename)
    
    print('Loading data frame(s) from {} ...'.format(dir_path + filename), end='')
    with open(dir_path + filename, 'r') as reader:
        data_frame = pickle.load(reader)
    print('Complete')
    return data_frame