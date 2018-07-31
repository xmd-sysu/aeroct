'''
Created on Jun 22, 2018

@author: savis

TODO: Change aod_type from int to string (needs to be changed in match_up.py too)
TODO: Add doc-strings to the data-frame classes. 
'''
from __future__ import print_function, division
import os
from datetime import timedelta, datetime
import numpy as np
from scipy import stats
import pandas as pd
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import sys
sys.path.append('/home/h01/savis/workspace/summer')
from aeroct import aeronet
from aeroct import modis
from aeroct import metum
from aeroct import utils

SCRATCH_PATH = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/'

# How to output the names of the data sets
print_name = {'aeronet': 'AERONET',
              'modis': 'MODIS',
              'modis_t' : 'MODIS Terra',
              'modis_a' : 'MODIS Aqua',
              'metum': 'Unified Model'}


class DataFrame():
    '''
    The data frame into which the AOD data is processed. Only a single day's data is
    included from a single source (& forecast time).
    
    The total AOD data is stored in the first index of the 'aod' attribute. If there are
    values for the AOD due to dust at every location then these are stored in the second
    index of 'aod'. If there are not values for the dust AOD at every location (eg. MODIS
    data) then the total AOD values can be filtered using filters stored in the
    'dust_filters' attribute.
    
    The AOD data, longitudes, latitudes, and times for each data point are stored in 1D
    numpy arrays if the data frame is not loaded from an Iris cube (ie. not model data).
    If it has been loaded from an iris cube then the AOD data will be 3D and the
    longitudes, latitudes, and times attributes store only the axes data (the order for
    which is time, lat, lon).
    
    Initialising:
    
    Attributes:
    
    Methods:
    '''

    def __init__(self, aod, longitudes, latitudes, times, date, wavelength=550,
                 data_set=None, **kw):
        # Ensure longitudes are in range [-180, 180]
        longitudes = longitudes.copy()
        longitudes[longitudes > 180] -= 360
        
        # Data and axes
        self.aod        = aod                       # AOD data [Total, Dust])
        self.longitudes = longitudes                # [degrees]
        self.latitudes  = latitudes                 # [degrees]
        self.times      = times                     # [hours since 00:00:00 on date]
        
        # Meta-data
        self.date       = date                      # (datetime)
        self.wavelength = wavelength                # [nm]
        self.data_set   = data_set                  # The name of the data set
        self.forecast_time = kw.setdefault('forecast_time', None)  # [hours]
        self.name = print_name[data_set]
        if self.forecast_time is not None:
            self.name += ' (Lead time: {0} hours)'.format(int(self.forecast_time))
        
        # Dictionary containing lists of indices for the total AOD data which satisfies
        # various dust filter conditions. Only currently used for MODIS data. Eg:
        # Aerosol_Type_Land: Aerosol_Type_Land == 5
        # AE_Land:    Land AE < 0.6
        # SSA_Land:    0.878 > Single Scattering Albedo (470nm) > 0.955
        # FM_FRC_Ocean:    FM Fraction < 0.45
        # AE_Ocean:    Ocean AE < 0.6
        self.dust_filters = kw.setdefault('dust_filter', None)
        
        # Contains iris cube if from_cube() used
        self.cube = kw.setdefault('cube', None)
        
        self.additional_data = kw.setdefault('additional_data', [])
    
    
    @classmethod
    def from_cube(cls, cube, data_set):
        # Create a DataFrame using a cube containing model data (dust AOD only)
        aod = [None, cube.data]
        lons = cube.coord('longitude').points
        lats = cube.coord('latitude').points
        times = cube.coord('time').points
        
        date = cube.coord('date').points[0]
        wl = cube.coord('wavelength').points[0]
        fc_time = cube.coord('forecast_time').points[0]
        
        return cls(aod, lons, lats, times, date, wl, data_set,
                   forecast_time=fc_time, cube=cube)
    
    
    def datetimes(self):
        return [self.date + timedelta(hours=h) for h in self.times]
    
    
    def get_data(self, aod_type=None, dust_filter_fields=None):
        '''
        Get an array with either all the total AOD data or the dust AOD data. This is
        returned along with the corresponding longitudes, latitudes, and times.
        
        Parameters:
        aod_type : {None, 'total, or 'dust'}, optional (Default: None)
            The type of AOD data to return.
            None: Return the total AOD if the data frame contains both. If it contains
                  only one type of AOD data then that is returned instead.
            If 'total' or 'dust' is selected and the data frame does not contain that
            type of data then a ValueError is raised.
        dust_filter_fields : list of str, optional
            This is used if dust AOD is to be retrieved and the data frame does not
            contain dust AOD data at every location (ie. MODIS data). This lists the
            conditions to decide which AOD values are dominated by dust. The conditions
            within a secondary array are combined using AND, while the conditions in the
            first array are combined with OR. ie. [['a', 'b'], 'c'] represents
            (filter['a'] AND filter['b]) OR filter['c']. 
            MODIS fields:
            - 'ARSL_TYPE_LAND': If it has been flagged as dust already.
            - 'AE_LAND': angstrom exponent <= 0.6 for land data.
            - 'SSA_LAND': 0.878 < scattering albedo < 0.955 for land data.
            - 'FM_FRC_OCEAN': fine mode fraction <= 0.45 for ocean data.
            - 'AE_OCEAN': angstrom exponent <= 0.6 for ocean data.
            - 'REGION_OCEAN': Only ocean data within the regions with dust are selected.
            - 'NONE: No filter.
            By default ['ARSL_TYPE_LAND'] is used if the data has been retrieved from
            MetDB. If downloaded from NASA the following is used:
            [['AE_LAND', 'SSA_LAND'], ['FM_FRC_OCEAN', 'AE_OCEAN']]
        '''
        get_total = (aod_type=='total') | ((aod_type is None) &
                                           (self.aod[0] is not None))
        get_dust = (aod_type=='dust') | ((aod_type is None) &
                                         (self.aod[0] is None))
        
        # Total AOD selection
        if get_total:
            if self.aod[0] is None:
                raise ValueError('The data frame does not include total AOD data.')
            
            aod = self.aod[0]
            lon = self.longitudes
            lat = self.latitudes
            times = self.times
        
        # Dust AOD selection
        elif get_dust:
            if (self.aod[1] is None) & (self.dust_filters is None):
                raise ValueError('The data frame does not include dust AOD data.')
            
            if self.cube is None:
                
                # If all locations have dust AOD values
                if self.aod[1] is not None:
                    aod = self.aod[1]
                    lon = self.longitudes
                    lat = self.latitudes
                    times = self.times
                
                # Use the dust filter to decide which AOD values are dominated by dust
                else:
                    
                    # Get default filter fields
                    if dust_filter_fields is None:
                        if np.all(self.dust_filters['AE_LAND']):
                            dust_filter_fields = ['ARSL_TYPE_LAND']
                        else:
                            dust_filter_fields = [['AE_LAND', 'SSA_LAND'],
                                                  ['FM_FRC_OCEAN', 'AE_OCEAN']]
                    
                    # Perform AND over the filters in the second index and
                    # OR over the first index
                    dust_filter = []
                    for f in dust_filter_fields:
                        if isinstance(f, list):
                            dust_filter.append(np.all([self.dust_filters[f2]
                                                       for f2 in f], axis=0))
                        else:
                            dust_filter.append(self.dust_filters[f])
                    dust_filter = np.any(dust_filter, axis=0)
                    
                    aod = self.aod[0][dust_filter]
                    lon = self.longitudes[dust_filter]
                    lat = self.latitudes[dust_filter]
                    times = self.times[dust_filter]
                
            else:
                # If a cube is used then get the longitude, latitude and time for every point
                aod = self.aod[1].ravel()
                axes = np.ix_(self.times, self.latitudes, self.longitudes)
                grid = np.broadcast_arrays(*axes)
                lon = grid[2].ravel()
                lat = grid[1].ravel()
                times = grid[0].ravel()
        
        return aod, lon, lat, times
    
    
    def dump(self, filename=None, dir_path=SCRATCH_PATH+'data_frames/', filetype='pickle'):
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
        os.system('mkdir -p {0}'.format(dir_path))
        
        if filetype == 'pickle':
            file_ext = 'pkl'
        elif filetype == 'csv':
            file_ext = 'csv'
        
        if filename != None:
            pass
        elif type(self.data_set) == str:
            filename = '{0}_{1}_'.format(self.data_set, self.date.strftime('%Y%m%d'))
        else:
            raise ValueError('data_set attribute invalid. Cannot create filename')
        
        i = 0
        while os.path.exists(dir_path + filename + str(i).zfill(2) + file_ext):
            i += 1
        filepath = dir_path + filename + str(i).zfill(2) + file_ext
        
        # Write file
        os.system('touch {0}'.format(filepath))
        with open(filepath, 'w') as writer:
            pickle.dump(self, writer, -1)
        print('Data frame saved successfully to {0}'.format(filepath))
    
    
    def extract(self, lon_bounds=(-180, 180), lat_bounds=(-90, 90), time_bounds=(0, 24)):
        '''
        Return a new DataFrame only containing the data within the given bounds (inclusive).
        
        Parameters:
        lon_bounds : float tuple, optional (Default: (-180, 180))
            The bounds on the longitude.
        lat_bounds : float tuple, optional (Default: (-90, 90))
            The bounds on the latitude.
        time_bounds : float tuple, optional (Default: (0, 24))
            The bounds on the time (hours).
        '''
        in_lon = (self.longitudes >= lon_bounds[0]) & (self.longitudes <= lon_bounds[1])
        in_lat = (self.latitudes >= lat_bounds[0]) & (self.latitudes <= lat_bounds[1])
        in_time = (self.times >= time_bounds[0]) & (self.times <= time_bounds[1])
        in_bounds = (in_lon & in_lat & in_time)
        
        if self.cube is not None:
            lons = self.longitudes[in_lon]
            lats = self.latitudes[in_lat]
            times = self.times[in_time]
            cube = self.cube[in_time, in_lat, in_lon]
            
            if self.aod[0] is not None:
                aod0 = self.aod[0][in_time, in_lat, in_lon]
            else:
                aod0 = None
            if self.aod[1] is not None:
                aod1 = self.aod[1][in_time, in_lat, in_lon]
            else:
                aod1 = None
            
            if self.dust_filters is not None:
                dust_filters = self.dust_filters.copy()
                for key in dust_filters.keys():
                    dust_filters[key] = dust_filters[key][in_time, in_lat, in_lon]
            else:
                dust_filters = None
        
        else:
            lons = self.longitudes[in_bounds]
            lats = self.latitudes[in_bounds]
            times = self.times[in_bounds]
            cube = None
            
            if self.aod[0] is not None:
                aod0 = self.aod[0][in_bounds]
            else:
                aod0 = None
            if self.aod[1] is not None:
                aod1 = self.aod[1][in_bounds]
            else:
                aod1 = None
            
            if self.dust_filters is not None:
                dust_filters = self.dust_filters.copy()
                for key in dust_filters.keys():
                    dust_filters[key] = dust_filters[key][in_bounds]
            else:
                dust_filters = None
        
        ext_description = 'Extraction for lon: {0}, lat: {1}, time: {2}'\
                          .format(lon_bounds, lat_bounds, time_bounds)
        if hasattr(self, 'additional_data'):
            additional_data = self.additional_data.append(ext_description)
        else:
            additional_data = [ext_description]
        
        return DataFrame([aod0, aod1], lons, lats, times, self.date, self.wavelength,
                         self.data_set, forecast_time=self.forecast_time, cube=cube,
                         dust_filters=dust_filters, additional_data=additional_data)



class MatchFrame():
    '''
    The data frame into which the matched AOD data is processed for a single day.
    The averaged AOD data and standard deviations are contained within 2D numpy array,
    with the first index referring to the data set. The latitudes, longitudes, and times
    are all stored in 1D numpy arrays, all of the same length. The date, wavelength and
    forecast time (for models) are stored as attributes.
    '''

    def __init__(self, data, data_std, data_num, time_diff, longitudes, latitudes, times,
                 date, match_time, match_rad, wavelength=550, forecast_times=(None, None),
                 data_sets=(None, None), aod_type=0, **kw):
        # Data and axes
        self.data       = data              # Averaged AOD data (Not flattend if cube != None)
        self.data_std   = data_std          # Averaged AOD data standard deviations
        self.data_num   = data_num          # Number of values that are averaged
        self.time_diff  = time_diff         # The average difference in times (idx1 - idx0) 
        self.longitudes = longitudes        # [degrees]
        self.latitudes  = latitudes         # [degrees]
        self.times      = times             # [hours since 00:00:00 on date]
        
        # Meta-data
        self.date           = date            # (datetime)
        self.match_radius   = match_rad       # Maximum spacial difference between collocated points
        self.match_time     = match_time      # Maximum time difference between collocated points
        self.wavelength     = wavelength      # [nm]
        self.forecast_times = forecast_times  # [hours] tuple
        self.data_sets      = data_sets       # A tuple of the names of the data sets
        self.aod_type       = aod_type        # Whether it is coarse mode AOD or total AOD
        self.cube = kw.setdefault('cube', None)    # Contains AOD difference for a model-model match
        self.names = ['', '']
        for i in [0,1]:
            self.names[i] = print_name[data_sets[i]]
            if self.forecast_times[i] is not None:
                self.names[i] += ' (Lead time: {0} hours)'.format(int(self.forecast_times[i]))
        self.additional_data = kw.setdefault('additional_data', [])
        
        # Stats
        self.RMS = np.sqrt(np.mean((self.data[1] - self.data[0])**2))   # Root mean square
        self.BIAS_MEAN = np.mean(self.data[1] - self.data[0])           # y - x mean
        self.BIAS_STD = np.std(self.data[1] - self.data[0])             # standard deviation
        if self.data[0].size > 0:
            self.R_SLOPE, self.R_INTERCEPT, self.R = \
                stats.linregress(self.data[0], self.data[1])[:3]        # Regression and Pearson's correlation coefficient
        else:
            self.R_SLOPE, self.R_INTERCEPT, self.R = np.nan, np.nan, np.nan
    
    
    def datetimes(self):
        return [self.date + timedelta(hours=h) for h in self.times]
    
    def pd_dataframe(self):
        data_array = np.array([self.times, self.latitudes, self.longitudes,
                               self.data[0], self.data_std[0], self.data_num[0],
                               self.data[1], self.data_std[1], self.data_num[1],
                               self.time_diff]).T
        headers = ['Time', 'Latitude', 'Longitude',
                   '{0}: AOD average'.format(self.names[0]),
                   '{0}: AOD stdev'.format(self.names[0]),
                   '{0}: Number of points'.format(self.names[0]),
                   '{0}: AOD average'.format(self.names[1]),
                   '{0}: AOD stdev'.format(self.names[1]),
                   '{0}: Number of points'.format(self.names[1]),
                   'Average time difference']
        df = pd.DataFrame(data_array, columns=headers)
        return df
    
    
    def dump(self, filename=None, dir_path=SCRATCH_PATH+'match_frames/',
             filetype='pickle', overwrite=False):
        '''
        Save the data frame as a file in the chosen location. The filepath is returned.
        
        Parameters:
        filename: str, optional (Default: '{data_sets}_YYYYMMDD_##')
            What to name the saved file.
        dir_path: str, optional (Default: '/scratch/{USER}/aeroct/match_frames/')
            The path to the directory where the file will be saved.
        '''
        if dir_path[-1] != '/': dir_path += '/'
        
        # Make directory if it does not exist
        os.system('mkdir -p {0}'.format(dir_path))
        
        # File extension
        if filetype == 'pickle':
            file_ext = '.pkl'
        elif filetype == 'csv':
            file_ext = '.csv'
        
        if filename is None:
            
            if type(self.data_sets) == tuple:
                filename = '{0}-{1}_{2}_'.format(self.data_sets[1], self.data_sets[0],
                                              self.date.strftime('%Y%m%d'))
            else:
                raise ValueError('data_sets attribute invalid. Cannot create filename')
        
        i = 0
        while os.path.exists(dir_path + filename + str(i).zfill(2) + file_ext) & \
              (not overwrite):
            i += 1
        
        filepath = dir_path + filename + str(i).zfill(2) + file_ext
        os.system('touch {0}'.format(filepath))
        # Write pickle file
        if filetype == 'pickle':
            with open(filepath, 'w') as writer:
                pickle.dump(self, writer, -1)
        elif filetype == 'csv':
            df = self.pd_dataframe()
            df.to_csv(filepath)
        print('Data frame saved successfully to {0}'.format(filepath))
        
        return filepath
    
    def extract(self, lon_bounds=(-180, 180), lat_bounds=(-90, 90), time_bounds=(0, 24)):
        '''
        Return a new MatchFrame only containing the data within the given bounds (inclusive).
        
        Parameters:
        lon_bounds : float tuple, optional (Default: (-180, 180))
            The bounds on the longitude.
        lat_bounds : float tuple, optional (Default: (-90, 90))
            The bounds on the latitude.
        time_bounds : float tuple, optional (Default: (0, 24))
            The bounds on the time (hours).
        '''
        in_lon = (self.longitudes >= lon_bounds[0]) & (self.longitudes <= lon_bounds[1])
        in_lat = (self.latitudes >= lat_bounds[0]) & (self.latitudes <= lat_bounds[1])
        in_time = (self.times >= time_bounds[0]) & (self.times <= time_bounds[1])
        in_bounds = (in_lon & in_lat & in_time)
        
        if self.cube is not None:
            lons = self.longitudes[in_lon]
            lats = self.latitudes[in_lat]
            times = self.times[in_time]
            data = self.data[:, in_time, in_lat, in_lon]
            data_std = self.data_std[:, in_time, in_lat, in_lon]
            data_num = self.dat_num[:, in_time, in_lat, in_lon]
            time_diff = self.time_diff[in_time, in_lat, in_lon]
            cube = self.cube[in_time, in_lat, in_lon]
        
        else:
            lons = self.longitudes[in_bounds]
            lats = self.latitudes[in_bounds]
            times = self.times[in_bounds]
            data = self.data[:, in_bounds]
            data_std = self.data_std[:, in_bounds]
            data_num = self.data_num[:, in_bounds]
            time_diff = self.time_diff[in_bounds]
            cube = None
        
        ext_description = 'Extraction for lon: {0}, lat: {1}, time: {2}'\
                          .format(lon_bounds, lat_bounds, time_bounds)
        if hasattr(self, 'additional_data'):
            additional_data = self.additional_data.append(ext_description)
        else:
            additional_data = [ext_description]
        
        return MatchFrame(data, data_std, data_num, time_diff, lons, lats, times,
                          self.date, self.match_time, self.match_radius, self.wavelength,
                          self.forecast_times, self.data_sets, self.aod_type, cube=cube,
                          additional_data=additional_data)



def load(data_set, date, dl_dir=SCRATCH_PATH+'downloads/', forecast_time=0, src=None,
         dl_again=False):
    '''
    Load a data frame for a given date using data from either AERONET, MODIS, or the
    Unified Model (metum). This will allow it to be matched and compared with other data
    sets.
    
    Parameters:
    data_set: str
        The data set to load. This may be 'aeronet', 'modis', 'modis_a', 'modis_t',
        or 'metum'.
    date: str or datetime
        The date for the data that is to be loaded. Specify in format 'YYYYMMDD' for strings.
    dl_dir : str, optional (Default: '/scratch/{USER}/aeroct/downloads/')
        The directory in which to save downloaded data. The different data sets will be
        saved within directories in this location.
    forecast_time: int, optional (Default: 0)
        The forecast lead time to use if metum is chosen.
    src : str, optional (Default: None)
        The source to retrieve the data from.
        MODIS: None or 'NASA' to download from ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/
               'MetDB' for MetDB extraction (Note: fewer dust filters available)
    dl_again : bool, optional (Default: False)
        If it is True then it will download the data again, even if the file already
        exists.
    '''
    if dl_dir[-1] != '/':   dl_dir += '/'
    
    if isinstance(date, datetime):
        date = date.strftime('%Y%m%d')
    
    ds_names = {'aeronet': 'AERONET',
                'modis': 'MODIS',
                'modis_t' : 'MODIS_Terra',
                'modis_a' : 'MODIS_Aqua',
                'metum': 'Unified_Model'}
    ds_name = ds_names[data_set]
    
    if data_set == 'aeronet':
        print('-----------AERONET-----------')
        aer_dl_dir = dl_dir + 'AERONET/'
        filepath = '{0}AERONET_{1}'.format(aer_dl_dir, date)
        
        # Download data
        utils.download_range(data_set, [date], dl_dir=dl_dir, dl_again=dl_again)
        
        # Load downloaded data
        with open(filepath, 'r') as r:
            dl_data = pickle.load(r)
        
        aod_df = aeronet.parse_data(dl_data)
        print('Processing AERONET data...', end='')
        parameters = aeronet.process_data(aod_df, date)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set)
    
    elif data_set[:5] == 'modis':
        
        if data_set == 'modis_t':   satellite = 'Terra'
        elif data_set == 'modis_a': satellite = 'Aqua'
        else:                       satellite = 'Both'
        
        print('--------MODIS ({0})---------'.format(satellite))
        mod_dl_dir = dl_dir + 'MODIS/'
        filepath = '{0}{1}_{2}'.format(mod_dl_dir, ds_name, date)
        modis_filepath = '{0}MODIS_{1}'.format(mod_dl_dir, date)
        
        # Download data
        utils.download_range(data_set, [date], dl_dir=dl_dir, dl_again=dl_again)
        
        # Load downloaded data
        if os.path.exists(filepath):
            with open(filepath, 'r') as r:
                dl_data = pickle.load(r)
        elif os.path.exists(modis_filepath):
            with open(modis_filepath, 'r') as r:
                dl_data = pickle.load(r)
        
        print('Processing MODIS data...', end='')
        parameters = modis.process_data(dl_data, date, satellite, src=src)
        print('Complete.')
        return DataFrame(*parameters[:-1], data_set=data_set, dust_filter=parameters[-1])
    
    elif data_set == 'metum':
        print('------UNIFIED MODEL {0:03d}Z-----'.format(forecast_time))
        
        um_dl_dir = dl_dir + 'UM/'
        filepath = '{0}Unified_Model{1:03d}_{2}'.format(um_dl_dir, forecast_time, date)
        
        # Download data
        utils.download_range(data_set, [date], dl_dir, forecast_time, dl_again=dl_again)
        
        # Load the downloaded data
        with open(filepath, 'r') as r:
            aod_cube = pickle.load(r)
            
        print('Processing Unified Model data...', end='')
        aod_cube = metum.process_data(aod_cube, date, forecast_time)
        print('Complete.')
        return DataFrame.from_cube(aod_cube, data_set)
    
    else:
        raise ValueError('Invalid data set: {0}'.format(data_set))

def load_from_pickle(filename, dir_path=SCRATCH_PATH+'match_frames'):
    '''
    Load the data frame from a file in the chosen location. Note that saving and
    loading large data frames can take some time.
    
    Parameters:
    filename : str
        The name of the saved file.
    dir_path : str, optional (Default: '/scratch/{USER}/aeroct/match_frames/')
        The path to the directory from which to load the file.
    '''
    if dir_path[-1] != '/': dir_path += '/'
    
    if not os.path.exists(dir_path + filename):
        raise ValueError('File does not exist: {0}'.format(dir_path + filename))
    
    print('Loading data frame(s) from {0}...'.format(dir_path + filename), end='')
    with open(dir_path + filename, 'r') as reader:
        data_frame = pickle.load(reader)
    print('Complete')
    return data_frame


def concatenate_data_frames(df_list):
    '''
    Concatenate a list of data frames over a period of time so that the average may be
    plotted on a map. A data frame of the input type (DataFrame or MatchFrame)
    (Currently only works for MatchFrames) is returned with a date attribute containing
    the list of dates.
    
    Parameters:
    df_list : iterable of DataFrames / MatchFrames
        The list of data frames over a period. All must have the same wavelength and
        data-set(s). 
    '''
    # Currently only works for MatchFrames
    if isinstance(df_list[0], MatchFrame):
        
        match_time = df_list[0].match_time
        match_rad = df_list[0].match_radius
        wavelength = df_list[0].wavelength
        fc_times = df_list[0].forecast_times
        data_sets = df_list[0].data_sets
        aod_type = df_list[0].aod_type
        
        dates = []
        data0, data_std0, data_num0 = [], [], []
        data1, data_std1, data_num1 = [], [], []
        longitudes, latitudes, times = [], [], []
        time_diff = []
        
        for df in df_list:
            
            # Check that the wavelengths and data-sets all match
            if df.wavelength != df_list[0].wavelength:
                raise ValueError('The list of data frames do not contain data for the same\
                                  wavelength.')
            if df.data_sets != df_list[0].data_sets:
                raise ValueError('The list of data frames do not contain data from the\
                                  same data-sets.')
            
            dates.append(df.date)
            data0.extend(df.data[0])
            data_std0.extend(df.data_std[0])
            data_num0.extend(df.data_num[0])
            data1.extend(df.data[1])
            data_std1.extend(df.data_std[1])
            data_num1.extend(df.data_num[1])
            time_diff.extend(df.time_diff)
            longitudes.extend(df.longitudes)
            latitudes.extend(df.latitudes)
            times.extend(df.times)
        
        data = np.array([data0, data1])
        data_std = np.array([data_std0, data_std1])
        data_num = np.array([data_num0, data_num1])
        longitudes, latitudes = np.array(longitudes), np.array(latitudes)
        times = np.array(times)
        
        return MatchFrame(data, data_std, data_num, time_diff, longitudes, latitudes, times,
                          dates, match_time, match_rad, wavelength, fc_times,
                          data_sets, aod_type)
    
    if isinstance(df_list[0], DataFrame):
        
        wavelength = df_list[0].wavelength
        fc_time = df_list[0].forecast_time
        data_set = df_list[0].data_set
        
        dates, aod0, aod1, longitudes, latitudes, times = [], [], [], [], [], []
        
        for df in df_list:
            
            # Check that the wavelengths and data-sets all match
            if df.wavelength != df_list[0].wavelength:
                raise ValueError('The list of data frames do not contain data for the same\
                                  wavelength.')
            if df.data_set != df_list[0].data_set:
                raise ValueError('The list of data frames do not contain data from the\
                                  same data-sets.')
            
            dates.append(df.date)
            if df.aod[0] is None: aod0 = None
            else: aod0.extend(df.aod[0].ravel())
            if df.aod[1] is None: aod1 = None
            else: aod1.extend(df.aod[1].ravel())
            longitudes.extend(df.longitudes)
            latitudes.extend(df.latitudes)
            times.extend(times)
        
        if aod0 is not None: aod0 = np.array(aod0)
        if aod1 is not None: aod1 = np.array(aod1)
        aod = [aod0, aod1]
        longitudes, latitudes = np.array(longitudes), np.array(latitudes)
        times = np.array(times)
        
        return DataFrame(aod, longitudes, latitudes, times, dates, wavelength, data_set,
                         forecast_time=fc_time)



if __name__=='__main__':
    days = np.arange(47)
    initial_date = datetime(2018, 6, 1)
    dt_list = [initial_date + timedelta(days=int(d)) for d in days]
    
    for date in dt_list:
        print('Downloading for: ', date)
        load('metum', date.strftime('%Y%m%d'), forecast_time=6, dl_again=True)
        load('metum', date.strftime('%Y%m%d'), forecast_time=12)
        load('metum', date.strftime('%Y%m%d'), forecast_time=18)