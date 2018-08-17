'''
Created on Jun 22, 2018

@author: savis
'''
from __future__ import print_function, division
import os
from datetime import timedelta, datetime
import numpy as np
from scipy import stats
import pandas as pd
import warnings
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from aeroct import aeronet
from aeroct import modis
from aeroct import metum
from aeroct import utils

SCRATCH_PATH = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/'
MODIS_DUST_FILTERS = [['ARSL_TYPE_LAND'],['AE_LAND', 'SSA_LAND'],
                      ['FM_FRC_OCEAN', 'AE_OCEAN', 'EFF_RAD_OCEAN', 'MASS_CONC', 'REGION_OCEAN']]

# How to output the names of the data sets
ds_printname = {'aeronet': 'AERONET',
              'modis': 'MODIS',
              'modis_t' : 'MODIS Terra',
              'modis_a' : 'MODIS Aqua',
              'metum': 'Unified Model'}
ds_filename = {'aeronet': 'AERONET',
               'modis': 'MODIS',
               'modis_t' : 'MODIS_Terra',
               'modis_a' : 'MODIS_Aqua',
               'metum': 'Unified_Model'}

def set_dust_filters(dust_filters):
    '''
    Set the dust filters to use for retrieving dust data for MODIS.
    
    Parameters
    ----------
    dust_filters : list of str
        This lists the conditions to decide which AOD values are dominated by dust.
        The conditions within a secondary array are combined using AND, while the
        conditions in the first array are combined with OR.
        ie. [['a', 'b'], 'c'] represents (filter['a'] AND filter['b]) OR filter['c'].
        Available filters:
        - 'ARSL_TYPE_LAND': If it has been flagged as dust already.
        - 'AE_LAND': angstrom exponent <= 0.6 for land data.
        - 'SSA_LAND': 0.878 < scattering albedo < 0.955 for land data.
        - 'FM_FRC_OCEAN': fine mode fraction <= 0.45 for ocean data.
        - 'AE_OCEAN': angstrom exponent <= 0.5 for ocean data.
        - 'EFF_RAD_OCEAN' : effective radius > 1.0 micron.
        - 'MASS_CONC' : mass concentration >= 1.2e-4 kg / m^2.
        - 'REGION_OCEAN': Only ocean data within the dust regions are selected.
        - 'NONE': No filter.
        By default the following are used:
        ['ARSL_TYPE_LAND', ['AE_LAND', 'SSA_LAND'], ['FM_FRC_OCEAN', 'AE_OCEAN',
         'EFF_RAD_OCEAN', 'MASS_CONC', 'REGION_OCEAN']]
        However if the data has been retrieved from MetDB only ARSL_TYPE_LAND has an
        effect.
    '''
    global MODIS_DUST_FILTERS
    MODIS_DUST_FILTERS = dust_filters


class DataFrame():
    '''
    The data frame into which the AOD data is processed. Only a single day's data is
    included from a single source (& forecast time).
    
    The total AOD data is stored in the first index of the 'aod' attribute. If there are
    values for the AOD due to dust at every location then these are stored in the second
    index of 'aod'. If there are filters in the 'dust_filters' attribute (eg. MODIS data)
    then the AOD values can be filtered using various combinations of these to obtain the
    dust dominated data points.
    
    The AOD data, longitudes, latitudes, and times for each data point are stored in 1D
    NumPy arrays if the data frame is not loaded from an Iris cube (ie. not model data).
    If it has been loaded from an iris cube then the AOD data will be 3D and the
    longitudes, latitudes, and times attributes store only the axes data (the order for
    which is time, lat, lon).
    
    ----------
    Attributes
    ----------
    aod : list of two 1D or 3D NumPy arrays
        The first element of the list contains the total AOD, the second contains the
        coarse mode AOD. These are only 3D when the data is gridded in which case 'cube'
        contains an Iris cube. If 3D the order of indices is [time, latitude, longitude].
    longitudes : 1D NumPy array
        This contains the values of the longitude either at every data point or along an
        axis. (In range -180 to 180)
    latitudes : 1D NumPy array
        This contains the values of the latitude either at every data point or along an
        axis.
    times : 1D NumPy array
        This contains the values of the time in hours from 00:00 on the given date,
        either at every data point or along an axis.
    date : datetime
        The date for which the DataFrame instance contains data. Note that this is not
        the date of the beginning of the forecast if the DataFrame contains forecast
        data.
    wavelength : int
        The wavelength, in nm, for which the AOD data has been taken. (Usually 550 nm)
    forecast_time : float
        If the data is from a model this contains the forecast lead time in hours.
        Otherwise it is None.
    data_set : {'aeronet', 'metum', 'modis', 'modis_a', 'modis_t'}
        This indicates the source of the data contained within the DataFrame.
    name : str
        This gives the name of the source in a format for printing. It also includes the
        forecast lead time for model data.
    sites : str or None
        For AERONET data this contains all of the names of the sites in the data
    dust_filters : dict or None
        Dictionary containing lists of indices for the AOD data which satisfies
        various dust filter conditions. Only currently used for MODIS data.
        Possible MODIS fields:
            - 'ARSL_TYPE_LAND': If it has been flagged as dust already.
            - 'AE_LAND': Angstrom exponent <= 0.6 for land data.
            - 'SSA_LAND': 0.878 < scattering albedo < 0.955 for land data.
            - 'FM_FRC_OCEAN': Fine mode fraction <= 0.45 for ocean data.
            - 'AE_OCEAN': Angstrom exponent <= 0.5 for ocean data.
            - 'EFF_RAD_OCEAN': effective radius > 1.0 micron.
            - 'MASS_CONC': mass concentration >= 1.2e-4 kg / m^2.
            - 'REGION_OCEAN': Only ocean data within the regions with dust are selected.
            - 'NONE': No filter.
    cube : Iris cube or None
        If the data is obtained from an Iris cube then it is supplied here.
    additional_data : list of str
        Extra descriptive data about the data frame such as whether it has been
        extracted from another DataFrame and the bounds used.
    
    -------
    Methods
    -------
    datetimes :
        Returns the times as a list of datetime objects rather than the time in hours.
    get_data :
        Get an array with either all the total / dust AOD data or the dust AOD data using
        dust filters. This is returned along with the corresponding longitudes,
        latitudes, and times. 
    dump :
        Saves the DataFrame as a pickled file in the chosen location.
    extract :
        Return a new DataFrame containing only the data within the given bounds (inclusive).
    
    ------------
    Initialising
    ------------
    Parameters for calling the class directly:
    aod : list of two 1D or 3D NumPy arrays
        The first element of the list contains the total AOD, the second contains the
        coarse mode AOD. These should only be 3D when a cube is passed as an argument;
        in this case the order of indices is [time, latitude, longitude].
    longitudes : 1D NumPy array
        This contains the values of the longitude either at every data point or along an
        axis. (In range -180 to 180)
    latitudes : 1D NumPy array
        This contains the values of the latitude either at every data point or along an
        axis.
    times : 1D NumPy array
        This contains the values of the time in hours from 00:00 on the given date,
        either at every data point or along an axis.
    date : datetime
        The date for which the DataFrame instance contains data. Note that this is not
        the date of the beginning of the forecast if the DataFrame contains forecast
        data.
    wavelength : int, optional (Default: 550)
        This is the wavelength in nm at which the AOD data has been taken.
    data_set : {'aeronet', 'metum', 'modis', 'modis_a', 'modis_t'}
        This indicates the source of the data contained within the DataFrame.
    Optional kwargs:
    forecast_time : float, optional (Default: None)
        If the data is from a model this contains the forecast lead time in hours.
    dust_filters : dict, optional (Default: None)
        A dictionary containing lists of indices for the AOD data which satisfies
        various dust filter conditions.
    cube : Iris cube (Default: None)
        If the data is obtained from an Iris cube then it is supplied here.
    additional_data : list of str (Default: [])
        Extra descriptive data about the data frame.
    
    Parameters for the from_cube class method:
    cube : Iris cube
        This is the Iris cube that contains the AOD data required to create the
        DataFrame.
    data_set : {'metum'}
        This indicates the source of the data.
    '''

    def __init__(self, aod, longitudes, latitudes, times, date, wavelength=550,
                 data_set=None, **kwargs):
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
        self.forecast_time = kwargs.setdefault('forecast_time', None)  # [hours]
        # Name for printing
        self.name = ds_printname[data_set]
        if self.forecast_time is not None:
            self.name += ' (T+{0}h)'.format(int(self.forecast_time))
        
        self.sites = kwargs.setdefault('sites', None)
        self.dust_filters = kwargs.setdefault('dust_filters', None)
        self.cube = kwargs.setdefault('cube', None)
        self.additional_data = kwargs.setdefault('additional_data', [])
    
    
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
        Return: aod, lon, lat, times.
        
        Parameters
        ----------
        aod_type : {None, 'total, or 'dust'}, optional (Default: None)
            The type of AOD data to return.
            None: Return the total AOD if the data frame contains both. If it contains
                  only one type of AOD data then that is returned instead.
            If 'total' or 'dust' is selected and the data frame does not contain that
            type of data then a ValueError is raised.
        dust_filter_fields : list of str, optional
            This is used if dust AOD is to be retrieved and the data frame contains dust
            filters (ie. MODIS data). This lists the conditions to decide which AOD
            values are dominated by dust. The conditions within a secondary array are
            combined using AND, while the conditions in the first array are combined with OR.
            ie. [['a', 'b'], 'c'] represents (filter['a'] AND filter['b]) OR filter['c'].
            MODIS fields:
            - 'ARSL_TYPE_LAND': If it has been flagged as dust already.
            - 'AE_LAND': angstrom exponent <= 0.6 for land data.
            - 'SSA_LAND': 0.878 < scattering albedo < 0.955 for land data.
            - 'FM_FRC_OCEAN': fine mode fraction <= 0.45 for ocean data.
            - 'AE_OCEAN': angstrom exponent <= 0.5 for ocean data.
            - 'EFF_RAD_OCEAN' : effective radius > 1.0 micron.
            - 'MASS_CONC' : mass concentration >= 1.2e-4 kg / m^2.
            - 'REGION_OCEAN': Only ocean data within the dust regions are selected.
            - 'NONE': No filter.
            By default ['ARSL_TYPE_LAND'] is used if the data has been retrieved from
            MetDB. If downloaded from NASA the following is used:
            [['AE_LAND', 'SSA_LAND'], ['FM_FRC_OCEAN', 'AE_OCEAN', 'EFF_RAD_OCEAN',
                                       'MASS_CONC', 'REGION_OCEAN']]
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
                
                # If there are no dust filters
                if self.dust_filters is None:
                    aod = self.aod[1]
                    lon = self.longitudes
                    lat = self.latitudes
                    times = self.times
                
                # Use the dust filter to decide which AOD values are dominated by dust
                else:
                    
                    # Get default filter fields
                    if dust_filter_fields is None:
                        # From MetDB
                        if np.all(self.dust_filters['AE_LAND']):
                            if MODIS_DUST_FILTERS != ['NONE']:
                                dust_filter_fields = ['ARSL_TYPE_LAND']
                            else:
                                dust_filter_fields = ['NONE']
                        # From NASA
                        else:
                            dust_filter_fields = MODIS_DUST_FILTERS
                    
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
                    
                    if self.aod[1] is None:
                        aod = self.aod[0][dust_filter]
                    else:
                        aod = self.aod[1][dust_filter]
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
    
    
    def dump(self, filename=None, save_dir=SCRATCH_PATH+'data_frames/'):
        '''
        Saves the DataFrame as a pickled file in the chosen location. Note that some
        DataFrames can be very large and take some time to save / load.
        
        Parameters
        ----------
        filename : str, optional (Default: '{data_set}_YYYYMMDD')
            What to name the saved file.
        save_dir : str, optional (Default: '/scratch/{USER}/aeroct/data_frames/')
            The path to the directory where the file will be saved.
        '''
        # Make directory if it does not exist
        os.system('mkdir -p {0}'.format(save_dir))
        file_ext = 'pkl'
        
        if filename != None:
            pass
        elif type(self.data_set) == str:
            if self.forecast_time is None:
                filename = '{0}_{1}'.format(self.data_set, self.date.strftime('%Y%m%d'))
            else:
                filename = '{0}{1}_{2}'.format(self.data_set, self.forecast_time,
                                               self.date.strftime('%Y%m%d'))
        else:
            raise ValueError('data_set attribute invalid. Cannot create filename')
        
        # Write file
        filepath = save_dir + filename + file_ext
        os.system('touch {0}'.format(filepath))
        with open(filepath, 'w') as writer:
            pickle.dump(self, writer, -1)
        print('Data frame saved successfully to {0}'.format(filepath))
    
    
    def extract(self, bounds=(-180, 180, -90, 90), time_bounds=(0, 24)):
        '''
        Return a new DataFrame only containing the data within the given bounds (inclusive).
        
        Parameters
        ----------
        bounds : 4-tuple, or list of 4-tuples, optional (Default: (-180, 180, -90, 90))
            This contains the If it is a list of 4-tuples then each corresponds to a
            region for which the data shall be extracted. The 4-tuples contain the bounds
            as follows: (min lon, max lon, min lat, max lat)
        time_bounds : 2-tuple of floats, optional (Default: (0, 24))
            The bounds on the time (hours).
        '''
        if isinstance(bounds[0], (int, long, float)):
            in_lon = (self.longitudes >= bounds[0]) & (self.longitudes <= bounds[1])
            in_lat = (self.latitudes >= bounds[2]) & (self.latitudes <= bounds[3])
            in_bounds = np.array(in_lon & in_lat)
        
        else:
            in_bounds = np.zeros_like(self.longitudes)
            for bound in bounds:
                in_lon = (self.longitudes >= bound[0]) & (self.longitudes <= bound[1])
                in_lat = (self.latitudes >= bound[2]) & (self.latitudes <= bound[3])
                in_bounds += (in_lon & in_lat)
            in_bounds = np.array(in_bounds, dtype=bool)
        
        in_time = (self.times >= time_bounds[0]) & (self.times <= time_bounds[1])
        in_bounds = np.array(in_bounds & in_time)
        
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
            
            sites = self.sites[in_bounds] if (self.sites is not None) else None
            
            if self.dust_filters is not None:
                dust_filters = self.dust_filters.copy()
                for key in dust_filters.keys():
                    dust_filters[key] = dust_filters[key][in_bounds]
            else:
                dust_filters = None
        
        ext_description = 'Extraction for lon, lat: {0}, time: {1}'\
                          .format(bounds, time_bounds)
        if hasattr(self, 'additional_data'):
            additional_data = list(self.additional_data)
            additional_data.append(ext_description)
        else:
            additional_data = [ext_description]
        
        return DataFrame([aod0, aod1], lons, lats, times, self.date, self.wavelength,
                         self.data_set, forecast_time=self.forecast_time, sites=sites,
                         dust_filters=dust_filters, cube=cube,
                         additional_data=additional_data)



class MatchFrame():
    '''
    This data frame is used to contain AOD data matched between two data sources. Each
    matched-up data point is obtained by taking the mean of the original data points
    within a maximum distance and time. Various stats for the matched-up data are also
    supplied.
    
    Several attributes including the 'data' attribute containing matched-up AOD data have
    two values for the first index corresponding to the two data-sets. When plotted on a
    scatter plot the first of these (data[0]) is put on the x-axis, and the second
    (data[1]) along the y-axis. Additionally, the AOD bias is calculated as:
    data[1] - data[0].
    
    ----------
    Attributes
    ----------
    data : list of two 1D or 3D NumPy arrays
        The two elements of the list contain the mean matched-up AOD data for the two
        data-sets. These are only 3D when the data is gridded in which case 'cube'
        contains an Iris cube. If 3D the order of indices is [time, latitude, longitude].
    data_std : list of two 1D or 3D NumPy arrays
        The standard deviations of the matched-up AOD at every point for each data-set.
    data_num : list of two 1D or 3D NumPy arrays
        The number of original data points used to obtain each matched-up AOD value for
        each data-set.
    time_diff : 1D or 3D NumPy array
        The mean time difference between the original data points for each matched-up
        AOD value.
    longitudes : 1D NumPy array
        This contains the values of the longitude either at every data point or along an
        axis. (In range -180 to 180)
    latitudes : 1D NumPy array
        This contains the values of the latitude either at every data point or along an
        axis.
    times : 1D NumPy array
        This contains the values of the time in hours from 00:00 on the data's date,
        either at every data point or along an axis.
    sites : NumPy array of str or None
        The names of the AERONET sites for each of the data points if AERONET data has
        been matched, otherwise it is None.
    date : datetime, or list of datetimes
        The date for which the MatchFrame instance contains data. It is a list if
        multiple days have been concatenated into a single MatchFrame.
    match_dist : int
        The maximum distance for which data has been matched and averaged in km.
    match_time : int
        The maximum time over which data has been matched and averaged in hours.
    wavelength : int
        The wavelength, in nm, for which the AOD data has been taken. (Usually 550 nm)
    forecast_times : 2-tuple of floats
        If the data is from a model this contains the forecast lead time in hours,
        otherwise it is None. eg. (None, 6) if the first data-set is not a forecast and
        the second has a lead time of six hours.
    aod_type : {'total' or 'dust'}
        The type of AOD data which has been matched.
    data_sets : 2-tuple of {'aeronet', 'metum', 'modis', 'modis_a', 'modis_t'}
        This indicates the source of each set of data contained within the MatchFrame.
    names : 2-tuple of str
        This gives the name of the source of each data-set in a format for printing. It
        also includes the forecast lead time for model data.
    cube : Iris cube or None
        If the two data-sets have Iris cubes then this contains a cube with the bias of
        the data points.
    additional_data : list of str
        Extra descriptive data about the data frame such as whether it has been
        extracted from another DataFrame and the bounds used.
    num : int
        The total number of matched-up data points.
    rms : float
        The root mean square value of the data: sqrt(mean((data[1] - data[0])**2)).
    bias_mean : float
        The mean bias between the data: mean(data[1] - data[0]).
    bias_std : float
        The standard deviation of the bias between the data: std(data[1] - data[0]).
    r2 : float
        The coefficient of determination for the correlation to the y=x line.
    r_intercept, r_slope : float
        The linear regression coefficients for the data. 
    r : float
        The Pearson's correlation coefficient for the linear regression.
    log_r_intercept, log_r_slope : float
        The regression coefficients when fitting to the log of the data.
        ie. log10(y) = log_r_intercept + log10(x) * log_r_slope
    log_r : float
        The Pearson's correlation coefficient for the logarithmic regression.
    
    -------
    Methods
    -------
    datetimes :
        Returns the times as a list of datetime objects rather than the time in hours.
        Only possible if the MatchFrame has not been concatenated.
    pd_dataframe :
        Returns a Pandas dataframe containing the data for every data point. It does not
        contain metadata such as the date and wavelength.
    dump :
        Saves the MatchFrame in the chosen location either as a pickle file or a csv.
    extract :
        Return a new MatchFrame containing only the data within the given bounds.
    
    ------------
    Initialising
    ------------
    Parameters:
    data : list of two 1D or 3D NumPy arrays
        The two elements of the list contain the mean matched-up AOD data for the two
        data-sets. These should only be 3D when the data is gridded in which case 'cube'
        should be assigned. If 3D the order of indices is [time, latitude, longitude].
    data_std : list of two 1D or 3D NumPy arrays
        The standard deviations of the matched-up AOD at every point for each data-set.
    data_num : list of two 1D or 3D NumPy arrays
        The number of original data points used to obtain each matched-up AOD value for
        each data-set.
    time_diff : 1D or 3D NumPy array
        The mean time difference between the original data points for each matched-up
        AOD value.
    longitudes : 1D NumPy array
        This contains the values of the longitude either at every data point or along an
        axis. (In range -180 to 180)
    latitudes : 1D NumPy array
        This contains the values of the latitude either at every data point or along an
        axis.
    times : 1D NumPy array
        This contains the values of the time in hours from 00:00 on the data's date,
        either at every data point or along an axis.
    date : datetime, or list of datetimes
        The date for which the MatchFrame instance contains data. It should be a list if
        multiple days have been concatenated into a single MatchFrame.
    match_time : int
        The maximum time over which data has been matched and averaged in hours.
    match_dist : int
        The maximum distance for which data has been matched and averaged in degrees.
    wavelength : int, optional (Default: 550)
        The wavelength, in nm, for which the AOD data has been taken.
    forecast_times : 2-tuple of floats, optional (Default: (None, None))
        If the data is from a model this contains the forecast lead time in hours,
        otherwise it is None. eg. (None, 6) if the first data-set is not a forecast and
        the second has a lead time of six hours.
    data_sets : 2-tuple of {'aeronet', 'metum', 'modis', 'modis_a', 'modis_t'}
        This indicates the source of each set of data contained within the MatchFrame.
    aod_type : {'total' or 'dust'}
        The type of AOD data which has been matched.
    Optional kwargs:
    sites : NumPy array of str
        The names of the AERONET sites for each of the data points if AERONET data has
        been matched.
    cube : Iris cube or None
        If the two data-sets are both gridded then this should be assigned to a cube
        containg bias data.
    additional_data : list of str
        Extra descriptive data about the data frame.
    '''

    def __init__(self, data, data_std, data_num, time_diff, longitudes, latitudes, times,
                 date, match_time, match_dist, wavelength=550, forecast_times=(None, None),
                 data_sets=(None, None), aod_type='total', **kw):
        # Data and axes
        self.data       = data              # Averaged AOD data (Not flattend if cube != None)
        self.data_std   = data_std          # Averaged AOD data standard deviations
        self.data_num   = data_num          # Number of values that are averaged
        self.time_diff  = time_diff         # The average difference in times (idx1 - idx0) 
        self.longitudes = longitudes        # [degrees]
        self.latitudes  = latitudes         # [degrees]
        self.times      = times             # [hours since 00:00:00 on date]
        self.sites = kw.setdefault('sites', None) # AERONET site names
        
        # Meta-data
        self.date           = date            # (datetime)
        self.match_dist     = match_dist      # Maximum spacial difference between collocated points (km)
        self.match_time     = match_time      # Maximum time difference between collocated points
        self.wavelength     = wavelength      # [nm]
        self.forecast_times = forecast_times  # [hours] tuple
        self.data_sets      = data_sets       # A tuple of the names of the data sets
        self.aod_type       = aod_type        # Whether it is coarse mode AOD or total AOD
        self.cube = kw.setdefault('cube', None)    # Contains AOD difference for a model-model match
        self.names = ['', '']
        for i in [0,1]:
            self.names[i] = ds_printname[data_sets[i]]
            if self.forecast_times[i] is not None:
                self.names[i] += ' (T+{0}h)'.format(int(self.forecast_times[i]))
        self.additional_data = kw.setdefault('additional_data', [])
        
        # Stats
        self.num = self.data[0].size
        # Suppress warnings from averaging over empty arrays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.rms = np.sqrt(np.mean((self.data[1] - self.data[0])**2))   # Root mean square
            self.bias_mean = np.mean(self.data[1] - self.data[0])           # y - x mean
            self.bias_std = np.std(self.data[1] - self.data[0])             # standard deviation
        
        if self.num > 1:
            # R2
            y_mean = np.mean(self.data[1])
            ss_tot = np.sum((self.data[1] - y_mean) ** 2)
            ss_res = np.sum((self.data[1] - self.data[0]) ** 2)
            self.r2 = 1 - ss_res / ss_tot
            
            data0 = self.data[0][(self.data[0] > 0) & (self.data[1] > 0)]
            data1 = self.data[1][(self.data[0] > 0) & (self.data[1] > 0)]
            if data0.size > 2:
                # Linear Regression
                self.r_slope, self.r_intercept, self.r = \
                    stats.linregress(self.data[0], self.data[1])[:3]
                
                # Log Regression
                self.log_r_slope, self.log_r_intercept, self.log_r = \
                    stats.linregress(np.log10(data0), np.log10(data1))[:3]
            else:
                self.r_slope, self.r_intercept, self.r = np.nan, np.nan, np.nan
                self.log_r_slope, self.log_r_intercept, self.log_r = np.nan, np.nan, np.nan
                
        else:
            self.r_slope, self.r_intercept, self.r = np.nan, np.nan, np.nan
            self.log_r_slope, self.log_r_intercept, self.log_r = np.nan, np.nan, np.nan
            self.r2 = np.nan
    
    
    def datetimes(self):
        '''
        Returns the times as a list of datetime objects rather than the time in hours.
        Only possible if the MatchFrame has not been concatenated.
        '''
        if isinstance(self.date, datetime):
            return [self.date + timedelta(hours=h) for h in self.times]
        else:
            raise Exception('The MatchFrame must not be concatenated to use this method.')
    
    
    def pd_dataframe(self):
        '''
        Returns a Pandas dataframe containing the data for every data point. It does not
        contain metadata such as the date and wavelength.
        '''
        data_array = [self.times, self.latitudes, self.longitudes,
                               self.data[0], self.data_std[0], self.data_num[0],
                               self.data[1], self.data_std[1], self.data_num[1],
                               self.time_diff]
        headers = ['Time (hours)', 'Latitude', 'Longitude',
                   '1: AOD average'.format(self.names[0]),
                   '1: AOD stdev'.format(self.names[0]),
                   '1: Number of points'.format(self.names[0]),
                   '2: AOD average'.format(self.names[1]),
                   '2: AOD stdev'.format(self.names[1]),
                   '2: Number of points'.format(self.names[1]),
                   'Average time difference']
        
        if self.sites is not None:
            headers.insert(1, 'AERONET site')
            data_array.insert(1, self.sites)
        
        df = pd.DataFrame(np.array(data_array).T, columns=headers)
        return df
    
    
    def dump(self, filename=None, save_dir=SCRATCH_PATH+'match_frames/',
             filetype='pickle', subdir=True, verb=True):
        '''
        Save the data frame as a file in the chosen location if the file already exists
        it will be overwritten. The filepath is returned. Note that only pickle files can
        be used to load the MatchFrame as the csv files do not contain all of the
        necessary metadata.
        
        Parameters
        ----------
        filename : str, optional (Default: '{dataset2}-{dataset1}-{aod-type}-YYYYMMDD')
            What to name the saved file.
        save_dir : str, optional (Default: '/scratch/{USER}/aeroct/match_frames/')
            The path to the directory where the file will be saved.
        filetype : {'pickle', 'csv'}, optional (Default: 'pickle')
            The type of file to save. This will add a file extension.
        subdir : bool, optional (Default: True)
            Whether to save the MatchFrames within sub-directories.
        verb : bool, optional (Default: True)
            If True then a message is printed to the console if it saves successfully.
        '''
        if save_dir[-1] != '/': save_dir += '/'
        
        # Put the forecast time onto the end of model data-set names for the filename
        data_set_names = []
        for i, data_set in enumerate(self.data_sets):
            if self.forecast_times[i] is not None:
                fc_str = str(int(self.forecast_times[i])).zfill(3)
            else:
                fc_str = ''
            data_set_names.append(data_set + fc_str)
        
        # Subdirectories
        if subdir:
            save_dir += '{0}-{1}-{2}/'.format(data_set_names[1], data_set_names[0],
                                              self.aod_type[0])
        
        # File extension
        if filetype == 'pickle':
            save_dir += 'pkl/'
            file_ext = '.pkl'
        elif filetype == 'csv':
            file_ext = '.csv'
        
        # Make directory if it does not exist
        os.system('mkdir -p {0}'.format(save_dir))
        
        # Create the filename
        if filename is None:
            filename = '{0}-{1}-{2}-{3}'.format(data_set_names[1], data_set_names[0],
                                        self.aod_type[0], self.date.strftime('%Y%m%d'))
        
        filepath = save_dir + filename + file_ext
        os.system('touch {0}'.format(filepath))
        
        # Write pickle file
        if filetype == 'pickle':
            with open(filepath, 'w') as fout:
                pickle.dump(self, fout, -1)
        
        # Write csv file
        elif filetype == 'csv':
            if isinstance(self.date, list):
                first_date = self.date[0].strftime('%Y-%m-%d')
                last_date = self.date[-1].strftime('%Y-%m-%d')
                date_str = '{0} to {1}'.format(first_date, last_date)
            else:
                date_str = self.date.strftime('%Y-%m-%d')
            
            df = self.pd_dataframe()
            metadata = pd.Series([('Match up for {0}'.format(date_str)),
                                  ('Data-set 1 : {0}'.format(self.names[0])),
                                  ('Data-set 2 : {0}'.format(self.names[1])),
                                  ('Match distance  (km): {0}'.format(self.match_dist)),
                                  ('Match time (minutes): {0}'.format(self.match_time)),
                                  ('AOD type : {0}'.format(self.aod_type)),
                                  ('Wavelength : {0}'.format(self.wavelength)),
                                  (''),
                                  ('Number of matches : {0}'.format(self.num)),
                                  ('RMS : {0:.04f}'.format(self.rms)),
                                  ('Bias (2-1) mean : {0:.04f}'.format(self.bias_mean)),
                                  ('Bias (2-1) std : {0:.04f}'.format(self.bias_std)),
                                  ('Regression intercept : {0:.04f}'.format(self.r_intercept)),
                                  ('Regression slope : {0:.04f}'.format(self.r_slope)),
                                  ('Pearson R : {0:.04f}'.format(self.r)),
                                  ('')])
            
            with open(filepath, 'w') as fout:
                metadata.to_csv(fout, index=False)
                df.to_csv(fout)
        
        if verb: print('Data frame saved successfully to {0}'.format(filepath))
        return filepath
    
    
    def extract(self, bounds=(-180, 180, -90, 90), time_bounds=(0, 24)):
        '''
        Return a new MatchFrame only containing the data within the given bounds
        (inclusive).
        
        Parameters
        ----------
        bounds : 4-tuple, or list of 4-tuples, optional (Default: (-180, 180, -90, 90))
            This contains the If it is a list of 4-tuples then each corresponds to a
            region for which the data shall be extracted. The 4-tuples contain the bounds
            as follows: (min lon, max lon, min lat, max lat)
        time_bounds : float 2-tuple, optional (Default: (0, 24))
            The bounds on the time (hours).
        '''
        if isinstance(bounds[0], (int, long, float)):
            in_lon = (self.longitudes >= bounds[0]) & (self.longitudes <= bounds[1])
            in_lat = (self.latitudes >= bounds[2]) & (self.latitudes <= bounds[3])
            in_bounds = np.array(in_lon & in_lat)
        
        else:
            in_bounds = np.zeros_like(self.longitudes)
            for bound in bounds:
                in_lon = (self.longitudes >= bound[0]) & (self.longitudes <= bound[1])
                in_lat = (self.latitudes >= bound[2]) & (self.latitudes <= bound[3])
                in_bounds += (in_lon & in_lat)
            in_bounds = np.array(in_bounds, dtype=bool)
        
        in_time = (self.times >= time_bounds[0]) & (self.times <= time_bounds[1])
        in_bounds = np.array(in_bounds & in_time)
        
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
        
        ext_description = 'Extraction for lon, lat: {0}, time: {1}'\
                          .format(bounds, time_bounds)
        if hasattr(self, 'additional_data'):
            additional_data = list(self.additional_data)
            additional_data.append(ext_description)
        else:
            additional_data = [ext_description]
        
        return MatchFrame(data, data_std, data_num, time_diff, lons, lats, times,
                          self.date, self.match_time, self.match_dist, self.wavelength,
                          self.forecast_times, self.data_sets, self.aod_type, cube=cube,
                          additional_data=additional_data)



def load(data_set, date, dl_dir=SCRATCH_PATH+'downloads/', forecast_time=0, src=None,
         dl_again=False, verb=True):
    '''
    Load a data frame for a given date using data from either AERONET, MODIS, or the
    Unified Model (metum). This will allow it to be matched and compared with other data
    sets. If the necessary downloaded data exists within 'dl_dir' then that shall be
    used, otherwise the data will be downloaded.
    
    Parameters
    ----------
    data_set: str
        The data set to load. This may be 'aeronet', 'modis', 'modis_a', 'modis_t',
        or 'metum'.
    date: str or datetime
        The date for the data that is to be loaded. Specify in format 'YYYYMMDD' for strings.
    dl_dir : str, optional (Default: '/scratch/{USER}/aeroct/downloads/')
        The directory in which to save downloaded data. The different data sets will be
        saved within directories in this location.
    forecast_time: int, optional (Default: 0)
        The forecast lead time to use if a model is chosen.
    src : str, optional (Default: None)
        The source to retrieve the data from.
        MODIS: 'NASA' or None to download from ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/
               'MetDB' for MetDB extraction (Note: fewer dust filters available)
    dl_again : bool, optional (Default: False)
        If it is True then it will download the data again, even if the file already
        exists.
    verb: bool, optional (Default: True)
        If True then the steps being undertaken are printed to the console.
    '''
    if dl_dir[-1] != '/':   dl_dir += '/'
    
    if isinstance(date, datetime):
        date = date.strftime('%Y%m%d')
    
    ds_name = ds_filename[data_set]
    
    if data_set == 'aeronet':
        if verb: print('-----------AERONET-----------')
        
        aer_dl_dir = dl_dir + 'AERONET/'
        filepath = '{0}AERONET_{1}'.format(aer_dl_dir, date)
        
        # Download data
        utils.download_range(data_set, [date], dl_dir=dl_dir, dl_again=dl_again)
        
        # Load downloaded data
        with open(filepath, 'r') as r:
            dl_data = pickle.load(r)
        
        aod_df = aeronet.parse_data(dl_data)
        if verb: print('Processing AERONET data ... ', end='')
        parameters = aeronet.process_data(aod_df, date)
        if verb: print('Complete.')
        
        return DataFrame(*parameters[:-2], data_set=data_set, sites=parameters[-1])
    
    elif data_set[:5] == 'modis':
        
        if data_set == 'modis_t':   satellite = 'Terra'
        elif data_set == 'modis_a': satellite = 'Aqua'
        else:                       satellite = 'Both'
        
        if verb: print('--------MODIS ({0})---------'.format(satellite))
        
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
        
        if verb: print('Processing MODIS data ... ', end='')
        parameters = modis.process_data(dl_data, date, satellite, src=src)
        if verb: print('Complete.')
        
        return DataFrame(*parameters[:-1], data_set=data_set, dust_filters=parameters[-1])
    
    elif data_set == 'metum':
        if verb: print('-----UNIFIED MODEL LT:{0:03d}----'.format(forecast_time))
        
        um_dl_dir = dl_dir + 'UM/'
        filepath = '{0}Unified_Model{1:03d}_{2}'.format(um_dl_dir, forecast_time, date)
        
        # Download data
        utils.download_range(data_set, [date], dl_dir, forecast_time, dl_again=dl_again)
        
        # Load the downloaded data
        with open(filepath, 'r') as r:
            aod_cube = pickle.load(r)
            
        if verb: print('Processing Unified Model data ... ', end='')
        aod_cube = metum.process_data(aod_cube, date, forecast_time)
        if verb: print('Complete.')
        
        return DataFrame.from_cube(aod_cube, data_set)
    
    else:
        raise ValueError('Invalid data set: {0}'.format(data_set))


def load_from_pickle(filename, dir_path=SCRATCH_PATH+'match_frames/pkl/'):
    '''
    Load the data frame from a file in the chosen location. Note that saving and
    loading large DataFrames can take some time.
    
    Parameters
    ----------
    filename : str
        The name of the saved file.
    dir_path : str, optional (Default: '/scratch/{USER}/aeroct/match_frames/pkl/')
        The path to the directory from which to load the file.
    '''
    if dir_path[-1] != '/': dir_path += '/'
    
    if not os.path.exists(dir_path + filename):
        raise ValueError('File does not exist: {0}'.format(dir_path + filename))
    
    print('Loading data frame(s) from {0}.'.format(dir_path + filename))
    with open(dir_path + filename, 'r') as reader:
        data_frame = pickle.load(reader)
    return data_frame


def concatenate_data_frames(df_list):
    '''
    Concatenate a list of data frames over a period of time so that the average may be
    plotted on a map. A data frame of the input type (DataFrame or MatchFrame) is
    returned with a date attribute containing the list of dates.
    
    Parameters
    ----------
    df_list : iterable of DataFrames / MatchFrames
        The list of data frames over a period. All must have the same wavelength and
        data-set(s).
    '''
    if isinstance(df_list[0], MatchFrame):
        
        match_time = df_list[0].match_time
        match_rad = df_list[0].match_dist
        wavelength = df_list[0].wavelength
        fc_times = df_list[0].forecast_times
        data_sets = df_list[0].data_sets
        aod_type = df_list[0].aod_type
        additional_data = df_list[0].additional_data
        
        dates = []
        data0, data_std0, data_num0 = [], [], []
        data1, data_std1, data_num1 = [], [], []
        longitudes, latitudes, times = [], [], []
        time_diff = []
        
        for df in df_list:
            
            # Check that the wavelengths and data-sets all match
            if df.wavelength != df_list[0].wavelength:
                raise ValueError('The list of data frames do not contain data for the same wavelength.')
            if df.data_sets != df_list[0].data_sets:
                raise ValueError('The list of data frames do not contain data from the same data-sets.')
            if df.additional_data != df_list[0].additional_data:
                raise ValueError('The list of data frames do not contain data with the same meta-data: \n{0}{1}'\
                                  .format(additional_data, df.additional_data))
            
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
        time_diff = np.array(time_diff)
        longitudes, latitudes = np.array(longitudes), np.array(latitudes)
        times = np.array(times)
        
        return MatchFrame(data, data_std, data_num, time_diff, longitudes, latitudes, times,
                          dates, match_time, match_rad, wavelength, fc_times,
                          data_sets, aod_type, additional_data=additional_data)
    
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
            times.extend(df.times)
        
        if aod0 is not None: aod0 = np.array(aod0)
        if aod1 is not None: aod1 = np.array(aod1)
        aod = [aod0, aod1]
        longitudes, latitudes = np.array(longitudes), np.array(latitudes)
        times = np.array(times)
        
        if isinstance(df_list[0].dust_filters, dict):
            dust_filters = pd.concat([pd.DataFrame.from_dict(df.dust_filters) for df in df_list])
        else:
            dust_filters = None
        
        return DataFrame(aod, longitudes, latitudes, times, dates, wavelength, data_set,
                         forecast_time=fc_time, dust_filters=dust_filters)
