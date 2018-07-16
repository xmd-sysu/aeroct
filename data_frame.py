'''
Created on Jun 22, 2018

@author: savis

'''
from __future__ import print_function, division
import os
from datetime import timedelta
import numpy as np
from scipy import stats
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import aeronet
import modis
import metum

SCRATCH_PATH = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/'

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

    def __init__(self, aod, latitudes, longitudes, times, date, wavelength=550,
                 data_set=None, **kw):
        # Ensure longitudes are in range [-180, 180]
        longitudes = longitudes.copy()
        longitudes[longitudes > 180] -= 360
        
        self.aod        = aod                       # AOD data [Total, Dust])
        self.longitudes = longitudes                # [degrees]
        self.latitudes  = latitudes                 # [degrees]
        self.times      = times                     # [hours since 00:00:00 on date]
        
        self.date       = date                      # (datetime)
        self.wavelength = wavelength                # [nm]
        self.data_set   = data_set                  # The name of the data set
        self.forecast_time = kw.setdefault('forecast_time', None)  # [hours]
        
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
        
        return cls(aod, lats, lons, times, date, wl, data_set,
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
            conditions to decide which AOD values are dominated by dust.
            MODIS fields:
            - 'ARSL_TYPE_LAND': If it has been flagged as dust already.
            - 'AE_LAND': angstrom exponent <= 0.6 for land data.
            - 'SSA_LAND': 0.878 < scattering albedo < 0.955 for land data.
            - 'FM_FRC_OCEAN': fine mode fraction <= 0.45 for ocean data.
            - 'AE_OCEAN': angstrom exponent <= 0.6 for ocean data.
            - 'NONE: No filter.
            By default 'ARSL_TYPE_LAND' is selected if the data has been retrieved from
            MetDB, otherwise 'AE_LAND', 'SSA_LAND', 'FM_FRC_OCEAN' & 'AE_OCEAN' are used.
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
                            dust_filter_fields = ['AE_LAND', 'SSA_LAND',
                                                  'FM_FRC_OCEAN', 'AE_OCEAN']
                        
                    dust_filters = np.array([self.dust_filters[f]
                                             for f in dust_filter_fields])
                    is_dust = np.prod(dust_filters, axis=0, dtype=np.bool)
                    aod = self.aod[0][is_dust]
                    lon = self.longitudes[is_dust]
                    lat = self.latitudes[is_dust]
                    times = self.times[is_dust]
                
            else:
                # If a cube is used then get the longitude, latitude and time for every point
                aod = self.aod[1].ravel()
                axes = np.ix_(self.times, self.latitudes, self.longitudes)
                grid = np.broadcast_arrays(*axes)
                lon = grid[2].ravel()
                lat = grid[1].ravel()
                times = grid[0].ravel()
        
        return aod, lon, lat, times
    
    
    def dump(self, filename=None, dir_path=SCRATCH_PATH+'data_frames/'):
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
                 data_sets=(None, None), aod_type='total', **kw):
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
        self.BIAS_STD = np.std(self.data_f[1] - self.data_f[0])             # standard deviation
        self.R_SLOPE, self.R_INTERCEPT, self.R = \
            stats.linregress(self.data_f[0], self.data_f[1])[:3]            # Regression and Pearson's correlation coefficient
    
    
    def datetimes(self):
        return [self.date + timedelta(hours=h) for h in self.times]
    
    
    def dump(self, filename=None, dir_path=SCRATCH_PATH+'data_frames/'):
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
        
        if filename is None:
            
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
         dl_save=True, dl_dir=SCRATCH_PATH+'downloads/'):
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
        MODIS: None or 'MetDB' for MetDB extraction (Note: fewer dust filters available)
               'NASA' to download from ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/
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
            if (dl_save is True) | (dl_save == 'f'):
                
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
            
            if (src is None) | (src == 'MetDB'):
                print('Extracting {} data from MetDB for {}.'.format(ds_name, date))
                aod_dict = modis.retrieve_data_day_metdb(date, satellite)
            
            elif src == 'NASA':
                print('Downloading {} data for {}.'.format(ds_name, date))
                aod_dict = modis.load_data_day(date, dl_dir=dl_dir+'MODIS_hdf/',
                                               satellite=satellite, keep_files=True)
            
            # Save data
            if (dl_save is True) | (dl_save == 'f'):
                
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
        parameters = modis.process_data(aod_dict, date, satellite, src=src)
        print('Complete.\n')
        return DataFrame(*parameters[:-1], data_set=data_set, dust_filter=parameters[-1])
    
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


def load_from_file(filename, dir_path=SCRATCH_PATH+'data_frames'):
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
