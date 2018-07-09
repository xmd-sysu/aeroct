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
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import aeronet
import modis
import metum

scratch_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/'

# How to output the names of the data sets
name = {'aeronet': 'AERONET', 'modis': 'MODIS', 'metum': 'Unified Model'}

div0 = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b!=0)


class DataFrame():
    '''
    The data frame into which the AOD data is processed. Only a single day's data and
    some from the days before and after are included.
    The AOD data, latitudes, longitudes, and times are all stored in 1D numpy arrays, all
    of the same length. The date, wavelength and forecast time (for models) are stored
    as attributes. So each forecast time and date requires a new instance.
    '''

    def __init__(self, aod, aod_d, latitudes, longitudes, times, date, wavelength=550,
                 forecast_time=None, data_set=None, cube=None):
        # Ensure longitudes are in range [-180, 180]
        longitudes = longitudes.copy()
        longitudes[longitudes > 180] -= 360
        
        self.aod = aod                      # Total AOD data
        self.aod_d = aod_d                  # Dust component of AOD data
        self.longitudes = longitudes        # [degrees]
        self.latitudes = latitudes          # [degrees]
        self.times = times                  # [hours since 00:00:00 on date]
        
        self.date = date                    # (datetime)
        self.wavelength = wavelength        # [nm]
        self.forecast_time = forecast_time  # [hours]
        self.data_set = data_set            # The name of the data set
        self.cube = cube                    # contains iris cube if the from cube class method has been used
    
    
    @classmethod
    def from_cube(cls, cube, data_set):
        aod_c = cube.data                       # Model data is only for dust
        lons = cube.coord('longitude').points
        lats = cube.coord('latitude').points
        times = cube.coord('time').points
        
        date = cube.coord('date').points[0]
        wl = cube.coord('wavelength').points[0]
        fc_time = cube.coord('forecast_time').points[0]
        
        return cls(None, aod_c, lats, lons, times, date, wl, fc_time, data_set, cube)                
    
    
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
                 data_sets=(None, None), aod_type=0, cube=None):
        self.data = data                    # Averaged AOD data
        self.data_std = data_std            # Averaged AOD data standard deviations
        self.data_num = data_num            # Number of values that are averaged
        self.longitudes = longitudes        # [degrees]
        self.latitudes = latitudes          # [degrees]
        self.times = times                  # [hours since 00:00:00 on date]
        
        self.date = date                    # (datetime)
        self.wavelength = wavelength        # [nm]
        self.forecast_times = forecast_times# [hours] tuple
        self.data_sets = data_sets          # A tuple of the names of the data sets
        self.aod_type = aod_type            # Whether it is coarse mode AOD or total AOD
        self.cube = cube                    # Has a grid in space and time? (forecast)
        self.match_radius = match_rad       # Maximum spacial difference between collocated points
        self.match_time = match_time        # Maximum time difference between collocated points
        
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
    
    
    def scatter_plot(self, stats=True, show=True, error=True):
        '''
        This is used to plot AOD data from two sources which have been matched-up on a
        scatter plot. The output are the stats if stats=True and the figure also if
        show=True.
        
        Parameters:
        stats: bool, optional (Default: True)
            Choose whether to show statistics on the plot.    
        show: bool, optional (Default: True)
            If True, the plot is shown otherwise the figure is passed as an output.    
        error: bool, optional (Default: True)
            If True, error bars for the standard deviations are included on the plot.
        '''
        if len(self.data_sets) != 2:
            raise ValueError, 'The data frame must be matched-up data from two data sets'
        
        fig, ax = plt.subplots()
        
        if np.any([(self.data_sets[i] == 'metum') &
                   (self.data_sets[1-i] == 'modis') for i in [0,1]]):
            point_fmt = 'r.'
        else:
            point_fmt = 'ro'
        
        if error == True:
            plt.errorbar(self.data_f[0], self.data_f[1], self.std_f[1], self.std_f[0],
                         point_fmt, ecolor='gray')
        else:
            plt.plot(self.data_f[0], self.data_f[1], point_fmt)
        
        # Regression line
        x = np.array([0, 10])
        y = self.R_INTERCEPT + x * self.R_SLOPE
        plt.plot(x, y, 'g:', lw=1, scalex=False, scaley=False)
        
        # y = x line
        plt.plot([0, 10], [0, 10], 'k--', scalex=False, scaley=False)
        
        # Title and axes 
        plt.title('Collocated AOD comparison on {0}'.format(self.date.date()))
        if self.forecast_times[0] != None:
            plt.xlabel('{} (forecast lead time: {} hours)'\
                       .format(name[self.data_sets[0]], int(self.forecast_times[0])))
        else:
            plt.xlabel(name[self.data_sets[0]])
        if self.forecast_times[1] != None:
            plt.ylabel('{} (forecast lead time: {} hours)'\
                       .format(name[self.data_sets[1]], int(self.forecast_times[1])))
        else:
            plt.ylabel(name[self.data_sets[1]])
        
        # Stats
        if stats == True:
            rms_str = 'RMS: {:.02f}'.format(self.RMS)
            plt.text(0.03, 0.94, rms_str, fontsize=12, transform=ax.transAxes)
            bias_mean_str = 'Bias mean: {:.02f}'.format(self.BIAS_MEAN)
            plt.text(0.03, 0.88, bias_mean_str, fontsize=12, transform=ax.transAxes)
            bias_std_str = 'Bias std: {:.02f}'.format(self.BIAS_STD)
            plt.text(0.03, 0.82, bias_std_str, fontsize=12, transform=ax.transAxes)
            r_str = 'Pearson R: {:.02f}'.format(self.R)
            plt.text(0.35, 0.94, r_str, fontsize=12, transform=ax.transAxes)
            slope_str = 'Slope: {:.02f}'.format(self.R_SLOPE)
            plt.text(0.35, 0.88, slope_str, fontsize=12, transform=ax.transAxes)
            intercept_str = 'Intercept: {:.02f}'.format(self.R_INTERCEPT)
            plt.text(0.35, 0.82, intercept_str, fontsize=12, transform=ax.transAxes)
        
        if show == True:
            plt.show()
            return
        else:
            return fig
    
    
    def map_plot(self, lat=(-90,90), lon=(-180,180), plot_type='pcolormesh', grid_size=2):
        '''
        This can be used to plot the daily average of the AOD either at individual sites
        or on a grid.
        
        Parameters:
        lat: tuple, optional (Default: (-90, 90))
            A tuple of the latitude bounds of the plot in degrees.
        lon: tuple, optional (Default: (-180, 180))
            A tuple of the longitude bounds of the plot in degrees.
        plot_type: str, optional (Default: 'grid')
            The type of plot to produce if it does not contain AERONET data. 'pcolormesh'
            or 'contourf' are possible. If it contains AERONET data then a scatterplot of
            the sites is plotted.
        grid_size: float, optional (Default: 1)
            The size of the grid squares in degrees if not using indivual sites
        '''
        
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.title('Daily AOD difference between {1} & {2} for {0}'\
                  .format(self.date.date(), self.data_sets[1], self.data_sets[0]))
        cmap='PuOr'
        
        # Find the data within the given bounds
        in_bounds = (self.longitudes > lon[0]) & (self.longitudes < lon[1]) & \
                    (self.latitudes > lat[0]) & (self.latitudes < lat[1])
        lons = self.longitudes[in_bounds]
        lats = self.latitudes[in_bounds]
        aod_diff = self.data[1, in_bounds] - self.data[0, in_bounds]
        
        # If AERONET is included plot the sites on a map
        if np.any([self.data_sets[i] == 'aeronet' for i in [0,1]]):
            # Get the list of data points at each location
            site_lons, i_site = np.unique(self.longitudes, return_index=True)
            site_lats = self.latitudes[i_site]
            in_sites = site_lons[:, np.newaxis] == self.longitudes
            # Average the AOD at each site and take std
            aod_site_avg = np.mean(aod_diff * in_sites, axis=1)
            aod_site_std = np.mean(aod_diff**2 * in_sites, axis=1) - aod_site_avg**2
            
            plt.scatter(site_lons, site_lats, c=aod_site_avg, #norm=colors.LogNorm(),
                        cmap=cmap, s=100)
            ax.coastlines()
            plt.colorbar(orientation='horizontal')
            plt.show()
            return
        
        # PLOT A GRID
        # The AOD data will be put on a grid. Firstly get the latitudes and
        # longitudes of the grid points
        lat_grid = np.arange((lat[0] + grid_size/2), lat[1], grid_size)
        lon_grid = np.arange((lon[0] + grid_size/2), lon[1], grid_size)
        
        # Now put the lons and lats onto the nearest grid points
        lon_ints = np.rint((lons - lon[0]) / grid_size + 0.5)
        lons = lon[0] + (lon_ints - 0.5) * grid_size
        lat_ints = np.rint((lats - lat[0]) / grid_size + 0.5)
        lats = lat[0] + (lat_ints - 0.5) * grid_size
        
        aod_grid_avg = np.zeros((lat_grid.size, lon_grid.size))
        aod_grid_std = np.zeros((lat_grid.size, lon_grid.size))
        
        for i_lat, lat in enumerate(lat_grid):
            # For each longitude find the data points at that position
            # Then average the data for each grid point
            bool_mat = (lons == lon_grid[:, np.newaxis]) & (lats == np.array(lat))
            aod_diff_grid = aod_diff * bool_mat
            aod_diff_grid = np.where(aod_diff_grid!=0, aod_diff_grid, np.nan)
            
            # Suppress warnings from averaging over empty arrays
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                aod_grid_avg[i_lat] = np.nanmean(aod_diff_grid, axis=1)
                aod_grid_std[i_lat] = np.nanstd(aod_diff_grid, axis=1)
                  
        if plot_type == 'pcolormesh':
            plt.pcolormesh(lon_grid.ravel(), lat_grid.ravel(), aod_grid_avg, cmap=cmap)
        elif plot_type == 'contourf':
            plt.contourf(lon_grid.ravel(), lat_grid.ravel(), aod_grid_avg, cmap=cmap)
          
        ax.coastlines()
        plt.colorbar(orientation='horizontal')
        plt.show()


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
    
    if data_set == 'aeronet':
        dl_dir = dl_dir + 'AERONET/'
        
        if (not os.path.exists('{}AERONET_{}'.format(dl_dir, date))) | (dl_save == 'f'):
            print('Downloading AERONET data for ' + date +'.')
            aod_string = aeronet.download_data_day(date)
            if (dl_save == True) | (dl_save == 'f'):
                os.system('mkdir {} 2> /dev/null'.format(dl_dir))
                os.system('touch {}AERONET_{}'.format(dl_dir, date))
                with open('{}AERONET_{}'.format(dl_dir, date), 'w') as w:
                    pickle.dump(aod_string, w, -1)
        else:
            with open('{}AERONET_{}'.format(dl_dir, date), 'r') as r:
                    aod_string = pickle.load(r)
        
        aod_df = aeronet.parse_data(aod_string)
        print('Processing AERONET data...', end='')
        parameters = aeronet.process_data(aod_df, date)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set)
    
    elif data_set == 'modis':
        dl_dir = dl_dir + 'MODIS/'
        
        if (not os.path.exists('{}MODIS_{}'.format(dl_dir, date))) | (dl_save == 'f'):
            print('Downloading MODIS data for ' + date +'.')
            aod_array = modis.retrieve_data_day(date)
            if (dl_save == True) | (dl_save == 'f'):
                os.system('mkdir {} 2> /dev/null'.format(dl_dir))
                os.system("touch '{}MODIS_{}'".format(dl_dir, date))
                with open('{}MODIS_{}'.format(dl_dir, date), 'w') as w:
                    pickle.dump(aod_array, w, -1)
        else:
            with open('{}MODIS_{}'.format(dl_dir, date), 'r') as r:
                    aod_array = pickle.load(r)
        
        print('Processing MODIS data...', end='')
        parameters = modis.process_data(aod_array, date)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set)
    
    elif data_set == 'metum':
        dl_dir = dl_dir + 'UM/'
        
        force = False
        if dl_save == 'f':
            force = True
        
        metum.download_data_day(date, forecast_time, dl_dir, force)
        print('Loading files.')
        aod_cube = metum.load_files(date, forecast_time, dl_dir)
        print('Processing Unified Model data...', end='')
        aod_cube = metum.process_data(aod_cube, date, forecast_time)
        print('Complete.')
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