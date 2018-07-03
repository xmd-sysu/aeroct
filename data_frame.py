'''
Created on Jun 22, 2018

@author: savis

TODO: Add an attribute for if the data frames require testing for nearby times or
    locations.
'''

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from scipy import stats
import os
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import aeronet
import modis
import metum

scratch_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/'

div0 = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b!=0)

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
        self.data_set = data_set            # The name of the data set
    
    def datetimes(self):
        return [self.date + timedelta(hours=h) for h in self.times]
    
    
    def dump(self, filename=None, path=scratch_path+'data_frames/'):
        '''
        Save the data frame as a file in the chosen location. Note that saving and
        loading large data frames can take some time.
        
        Parameters:
        filename: str, optional (Default: '{data_set}_YYYYMMDD_##')
            What to name the saved file.
        path: str, optional (Default: '/scratch/{USER}/aeroct/data_frames/')
            The path to the directory where the file will be saved.
            
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
        
    def map_plot(self, lat=(-90,90), lon=(-180,180), plot_type='grid', grid_size=1):
        '''
        This can be used to plot the daily average of the AOD either at individual sites
        or on a grid.
        
        Parameters:
        lat: tuple, optional (Default: (-90, 90))
            A tuple of the latitude bounds of the plot in degrees.
        lon: tuple, optional (Default: (-180, 180))
            A tuple of the longitude bounds of the plot in degrees.
        plot_type: str, optional (Default: 'grid')
            The type of plot to produce. 'sites' for a plot of individual sites, 'grid'
            for a meshgrid plot, 'contourf' for a filled contour plot.
        grid_size: float, optional (Default: 1)
            The size of the grid squares in degrees if not using indivual sites
        '''
        
        if self.grid == False:
            ax = plt.axes(projection=ccrs.PlateCarree())
            plt.title('Daily AOD for {} from {}'.format(self.date.date(), self.data_set))
            cmap='Greys'
            
            if plot_type == 'sites':
                # Get the list of data points at each location
                site_lons, i_site = np.unique(self.longitudes, return_index=True)
                site_lats = self.latitudes[i_site]
                in_sites = site_lons[:, np.newaxis] == self.longitudes
                # Average the AOD at each site and take std
                aod_site_avg = np.mean(self.data * in_sites, axis=1)
                aod_site_std = np.mean(self.data**2 * in_sites, axis=1) - aod_site_avg**2
                
                plt.scatter(site_lons, site_lats, c=aod_site_avg, norm=colors.LogNorm(),
                            cmap=cmap, s=100)
                ax.coastlines()
                plt.colorbar(orientation='horizontal')
                plt.show()
                return
            
            # The AOD data will be put on a grid. Firstly get the latitudes and
            # longitudes of the grid points
            lat_grid = np.arange(lat[0] + grid_size/2,  lat[1], grid_size)
            lon_grid = np.arange(lon[0] + grid_size/2, lon[1], grid_size)
            lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
            
            # Find if each point of data lies within each grid point. This is stored in a
            # boolean array with indices: latitude, longitude, data frame index.
            lat_grid_bounds = np.arange(lat[0], lat[1] + grid_size/10, grid_size)
            lon_grid_bounds = np.arange(lon[0], lon[1] + grid_size/10, grid_size)
            
            aod_grid_avg = np.zeros_like(lat_grid)
            aod_grid_std = np.zeros_like(lat_grid)
            in_lon_grid = (self.longitudes < lon_grid_bounds[1:, np.newaxis]) & \
                          (self.longitudes > lon_grid_bounds[:-1, np.newaxis])
            for i_lat in np.arange(lat_grid_bounds.size - 1):
                in_lat_ar = (self.latitudes < lat_grid_bounds[i_lat + 1]) & \
                            (self.latitudes > lat_grid_bounds[i_lat])
                in_grid = in_lat_ar * in_lon_grid
                grid_data = self.data * in_grid
                grid_data = np.where(grid_data!=0, grid_data, np.nan)
                
                # Take the average and standard deviation for each grid point
                aod_grid_avg[i_lat] = np.nanmean(grid_data, axis=1)
                aod_grid_std[i_lat] = np.nanmean(grid_data**2, axis=1) \
                                        - aod_grid_avg[i_lat]**2
            
            if plot_type == 'grid':
                plt.pcolormesh(lon_grid, lat_grid, aod_grid_avg, norm=colors.LogNorm(),
                            cmap=cmap)
            elif plot_type == 'contourf':
                plt.contourf(lon_grid, lat_grid, aod_grid_avg, norm=colors.LogNorm(),
                            cmap=cmap)
            
            ax.coastlines()
            plt.colorbar(orientation='horizontal')
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
        
        # Flattened
        self.data_f = np.array([self.data[0].ravel(), self.data[0].ravel()])
        self.std_f = np.array([self.data_std[0].ravel(), self.data_std[0].ravel()])
        
        # Stats
        self.RMS = np.sqrt(np.mean((self.data_f[1] - self.data_f[0])**2))   # Root mean square
        self.R_SLOPE, self.R_INTERCEPT, self.R = \
            stats.linregress(self.data_f[0], self.data_f[1])[:3]            # Regression and Pearson's correlation coefficient
        self.BIAS_MEAN = np.mean(self.data_f[1] - self.data_f[0])           # y - x mean
        self.BIAS_STD = np.std(self.data_f[1] - self.data_f[0])             # y - x standard deviation
    
    
    def datetimes(self):
        return [self.date + timedelta(hours=h) for h in self.times]
    
    
    def dump(self, filename=None, path=scratch_path+'data_frames/'):
        '''
        Save the data frame as a file in the chosen location. Note that saving and
        loading large data frames can take some time. The filename is returned.
        
        Parameters:
        filename: str, optional (Default: '{data_sets}_YYYYMMDD_##')
            What to name the saved file.
        path: str, optional (Default: '/scratch/{USER}/aeroct/data_frames/')
            The path to the directory where the file will be saved.
        '''
        # Make directory if it does not exist
        os.system('mkdir -p {}'.format(path))
        
        if filename == None:
            if type(self.data_sets) == tuple:
                filename = '{}-{}_{}_'.format(self.data_sets[1], self.data_sets[0],
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
        
        fig = plt.figure()
        
        if error == True:
            plt.errorbar(self.data_f[0], self.data_f[1], self.std_f[1], self.std_f[0],
                         'ro', ecolor='gray')
        else:
            plt.plot(self.data_f[0], self.data_f[1], 'ro')
        
        high = np.nanmax(self.data)
        low = np.nanmin(self.data)
        plt.plot([low, high], [low, high], 'k--')
        
        plt.title('Collocated AOD comparison between {1} & {2} for {0}'\
                      .format(self.date.date(), self.data_sets[1], self.data_sets[0]))
        plt.xlabel(self.data_sets[0])
        plt.ylabel(self.data_sets[1])
        plt.loglog()
        plt.xlim(low, high)
        plt.ylim(low, high)
        
#        # Stats
#        rms = self.rms()
#        bias_mean = self.bias_mean()
#        bias_std = self.bias_std()
#        plt.text(low + 0.01*diff, high - 0.1*diff, 'RMS: {:.02f}'.format(rms), fontsize=15)
        
        if show == True:
            plt.show()
            return
        else:
            return fig
    
    
    def map_plot(self, lat=(-90,90), lon=(-180,180), plot_type='grid', grid_size=1):
        '''
        This can be used to plot the daily average of the AOD either at individual sites
        or on a grid.
        
        Parameters:
        lat: tuple, optional (Default: (-90, 90))
            A tuple of the latitude bounds of the plot in degrees.
        lon: tuple, optional (Default: (-180, 180))
            A tuple of the longitude bounds of the plot in degrees.
        plot_type: str, optional (Default: 'grid')
            The type of plot to produce. 'sites' for a plot of individual sites, 'grid'
            for a meshgrid plot, 'contourf' for a filled contour plot.
        grid_size: float, optional (Default: 1)
            The size of the grid squares in degrees if not using indivual sites
        '''
        
        if self.grid == False:
            ax = plt.axes(projection=ccrs.PlateCarree())
            plt.title('Daily AOD difference between {1} & {2} for {0}'\
                      .format(self.date.date(), self.data_sets[1], self.data_sets[0]))
            cmap='PuOr'
            
            aod_diff = self.data[1] - self.data[0]
            
            if plot_type == 'sites':
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
            
            # The AOD data will be put on a grid. Firstly get the latitudes and
            # longitudes of the grid points
            lat_grid = np.arange(lat[0] + grid_size/2,  lat[1], grid_size)
            lon_grid = np.arange(lon[0] + grid_size/2, lon[1], grid_size)
            lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
            
            # Find if each point of data lies within each grid point. This is stored in
            # boolean arrays for each latitude. 1st index: lon, 2nd: df2 index
            lat_grid_bounds = np.arange(lat[0], lat[1] + grid_size/10, grid_size)
            lon_grid_bounds = np.arange(lon[0], lon[1] + grid_size/10, grid_size)
            
            aod_grid_avg = np.zeros_like(lat_grid)
            aod_grid_std = np.zeros_like(lat_grid)
            print(self.longitudes)
            in_lon_grid = (self.longitudes < lon_grid_bounds[1:, np.newaxis]) & \
                          (self.longitudes > lon_grid_bounds[:-1, np.newaxis])
            for i_lat in np.arange(lat_grid_bounds.size - 1):
                in_lat_ar = (self.latitudes < lat_grid_bounds[i_lat + 1]) & \
                            (self.latitudes > lat_grid_bounds[i_lat])
                in_grid = in_lat_ar * in_lon_grid
                grid_data = aod_diff * in_grid
                grid_data = np.where(grid_data!=0, grid_data, np.nan)
                
                # Take the average and standard deviation for each grid point
                aod_grid_avg[i_lat] = np.nanmean(grid_data, axis=1)
                aod_grid_std[i_lat] = np.nanmean(grid_data**2, axis=1) \
                                        - aod_grid_avg[i_lat]**2
            
            if plot_type == 'grid':
                plt.pcolormesh(lon_grid, lat_grid, aod_grid_avg, #norm=colors.LogNorm(),
                            cmap=cmap)
            elif plot_type == 'contourf':
                plt.contourf(lon_grid, lat_grid, aod_grid_avg, #norm=colors.LogNorm(),
                            cmap=cmap)
            
            ax.coastlines()
            plt.colorbar(orientation='horizontal')
            plt.show()


def load(data_set, date, forecast_time=0, src=None, dir_path=scratch_path+'downloads/', save=True):
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
    src: str, optional (Default: None)
        The source to retrieve the data from.
        (Currently unavailable)
    dir_path: str, optional (Default: '/scratch/{USER}/aeroct/downloads/')
        The directory in which to save downloaded data.
    save: bool, optional (Default: True)
        Choose whether to save any downloaded data.
    '''
    if dir_path[-1] != '/':
        dir_path = dir_path + '/'
    
    if data_set == 'aeronet':
        if not os.path.exists('{}AERONET_{}'.format(dir_path, date)):
            print('Downloading AERONET data for ' + date +'.')
            aod_string = aeronet.download_data_day(date)
            if save == True:
                os.system('mkdir {} 2> /dev/null'.format(dir_path))
                os.system('touch {}AERONET_{}'.format(dir_path, date))
                with open('{}AERONET_{}'.format(dir_path, date), 'w') as w:
                    pickle.dump(aod_string, w, -1)
        else:
            with open('{}AERONET_{}'.format(dir_path, date), 'r') as r:
                    aod_string = pickle.load(r)
        
        aod_df = aeronet.parse_data(aod_string)
        print('Processing AERONET data...', end='')
        parameters = aeronet.process_data(aod_df, date)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set)
    
    elif data_set == 'modis':
        if not os.path.exists('{}MODIS_{}'.format(dir_path, date)):
            print('Downloading MODIS data for ' + date +'.')
            aod_array = modis.retrieve_data_day(date)
            if save == True:
                os.system('mkdir {} 2> /dev/null'.format(dir_path))
                os.system("touch '{}MODIS_{}'".format(dir_path, date))
                with open('{}MODIS_{}'.format(dir_path, date), 'w') as w:
                    pickle.dump(aod_array, w, -1)
        else:
            with open('{}MODIS_{}'.format(dir_path, date), 'r') as r:
                    aod_array = pickle.load(r)
        
        print('Processing MODIS data...', end='')
        parameters = modis.process_data(aod_array, date)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set)
    
    elif data_set == 'metum':
        metum.download_data_day(date, forecast_time, dir_path)
        print('Loading files.')
        aod_cube = metum.load_files(date, forecast_time, dir_path)
        print('Processing Unified Model data...', end='')
        parameters = metum.process_data(aod_cube, date, forecast_time)
        print('Complete.')
        return DataFrame(*parameters, data_set=data_set, grid=True)
    
    else:
        print('Invalid data set: {}'.format(data_set))
        return


def load_from_file(filename, dir_path=scratch_path+'data_frames'):
    '''
    Load the data frame from a file in the chosen location. Note that saving and
    loading large data frames can take some time.
    
    Parameters:
    filename: str
        The name of the saved file.
    path: str, optional (Default: '/scratch/{USER}/aeroct/data_frames/')
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