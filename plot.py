'''
This module contains the functions to plot single DataFrames or MatchFrames, ie. a single
day.
The AOD data throughout the day for a single AERONET site can be plotted with
plot_anet_site().
The daily average for the AOD / AOD difference for either a DataFrame or MatchFrame can
be plotted on a map using plot_map().
The AOD match-up for a MatchFrame may be plotted on a scatter plot with scatter_plot().

Created on Jul 5, 2018

@author: savis

TODO: Move the MatchFrame scatter plot function to this module.
'''
import sys
import numpy as np
from matplotlib import pyplot as plt, colors
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
sys.path.append('/home/h01/savis/workspace/summer')
import aeroct

# Suppress warnings from importing iris.plot in python 2
import warnings
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=MatplotlibDeprecationWarning)
    from iris import plot as iplt, analysis


def plot_anet_site(df, site=0):
    '''
    Plot the daily AOD data for a single AERONET site.
    
    Parameters:
    df : aeroct DataFrame
        A data-frame returned by aeroct.load() containing AERONET data, from which to plot.
    site : int, optional (Default: 0)
        The index for the AERONET site in a list sorted by increasing longitude.
    '''
    
    if df.data_set != 'aeronet':
        raise ValueError, 'Only AERONET data may be used in this function.'
    
    lons, i_uniq, i_inv = np.unique(df.longitudes, return_index=True , return_inverse=True)
    lon = lons[site]
    lat = df.latitudes[i_uniq[site]]
    
    aod = df.data[i_inv == site]
    times = df.times[i_inv == site]
    
    plt.plot(times, aod)#, 'ro')
    plt.title('Daily AOD from AERONET at (lon: {:.02f}, lat: {:.02f})'.format(lon, lat))
    plt.show()
    

def plot_map(df, lat=(-90,90), lon=(-180,180), plot_data=None, plot_type='pcolormesh',
             show=True, grid_size=0.5):
    '''
    This can be used to plot the daily average of the AOD either at individual sites
    or on a grid.
    
    Parameters:
    lat : tuple, optional (Default: (-90, 90))
        A tuple of the latitude bounds of the plot in degrees.
    lon : tuple, optional (Default: (-180, 180))
        A tuple of the longitude bounds of the plot in degrees.
    plot_type : str, optional (Default: 'scatter')
        The type of plot to produce. 'sites' for a plot of individual sites for AERONET
        data, 'scatter' to plot a scatter grid of all AOD data, 'pcolormesh' or
        'contourf' for griddded plots.
    show : bool, optional (Default: True)
        If True the figure is shown, otherwise it is returned 
    grid_size : float, optional (Default: 1)
        The size of the grid squares in degrees if not using indivdual sites
    '''
    
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.xlim(lon)
    plt.ylim(lat)
    
    data_frame_cmap = 'Greys'
    match_frame_cmap = 'RdBu'
    
    # USE IRIS PLOT IF THERE IS A CUBE IN THE DATA FRAME
    if df.cube != None:
        
        if df.__class__.__name__ == 'DataFrame':
            cmap = data_frame_cmap
            plt.title('Daily AOD for {} from {}'.format(df.date.date(), df.data_set))
        elif df.__class__.__name__ == 'MatchFrame':
            cmap = match_frame_cmap
            plt.title('Daily AOD difference : {1} - {2} for {0}'\
                  .format(df.date.date(), df.data_sets[1], df.data_sets[0]))
        
        print(df.cube)
        day_avg_cube = df.cube.collapsed('time', analysis.MEAN)
        
        if plot_type == 'pcolormesh':
            iplt.pcolormesh(day_avg_cube, cmap=cmap)
        if plot_type == 'contourf':
            iplt.contourf(day_avg_cube, cmap=cmap)
        
        ax.coastlines()
        plt.colorbar(orientation='horizontal')
        
        if show == True:
            plt.show()
            return
        else:
            return fig
    
    if df.__class__.__name__ == 'DataFrame':
        plt.title('Daily AOD for {} from {}'.format(df.date.date(), df.data_set))
        
        if (plot_data == None) | (plot_data == 'aod'):
            plot_data = df.aod
        elif plot_data == 'aod_d':
            plot_data = df.aod_d    # This will not work properly for MODIS data
        elif plot_data == 'times':
            plot_data = df.times
        
        # PLOT AOD AT AERONET SITES AS A SCATTER PLOT
        if df.data_set == 'aeronet':
            # Get the list of data points at each location
            site_lons, i_site = np.unique(df.longitudes, return_index=True)
            site_lats = df.latitudes[i_site]
            in_sites = site_lons[:, np.newaxis] == df.longitudes
            # Average the AOD at each site and take std
            aod_site_avg = np.mean(plot_data * in_sites, axis=1)
#             aod_site_std = np.mean(df.data**2 * in_sites, axis=1) - aod_site_avg**2
            
            plt.scatter(site_lons, site_lats, c=aod_site_avg, cmap=data_frame_cmap, s=100)
            ax.coastlines()
            plt.colorbar(orientation='horizontal')
            
            if show == True:
                plt.show()
                return
            else:
                return fig
        
        # PLOT THE AOD AT EVERY DATA POINT
        elif plot_type == 'scatter':
            plt.scatter(df.longitudes, df.latitudes, c=df.aod,
                        marker='o', s=(72./fig.dpi)**2)
            ax.coastlines()
            plt.colorbar(orientation='horizontal')
            
            if show == True:
                plt.show()
                return
            else:
                return fig
        
        # OTHERWISE PLOT A GRID
        # Using scipy.interpolate.griddata
        # First get the axes
        grid = np.mgrid[(lon[0] + grid_size/2) : lon[1] : grid_size,
                                      (lat[0] + grid_size/2) : lat[1] : grid_size]
        
        ll = zip(df.longitudes, df.latitudes)
        aod_grid = griddata(ll, plot_data, tuple(grid), method='linear')
        
        # Mask grid data where there are no nearby points. Firstly create kd-tree
        THRESHOLD = 2 * grid_size   # Maximum distance to look for nearby points
        tree = cKDTree(ll)
        xi = _ndim_coords_from_arrays(tuple(grid))
        dists = tree.query(xi)[0]
        # Copy original result but mask missing values with NaNs
        aod_grid[dists > THRESHOLD] = np.nan
        
        if plot_type == 'pcolormesh':
            plt.pcolormesh(grid[0], grid[1], aod_grid, cmap=data_frame_cmap)
        elif plot_type == 'contourf':
            plt.contourf(grid[0], grid[1], aod_grid, cmap=data_frame_cmap)
    
    elif df.__class__.__name__ == 'MatchFrame':
        plt.title('Daily AOD difference : {1} - {2} for {0}'\
                  .format(df.date.date(), df.data_sets[1], df.data_sets[0]))
        
        # Find the data within the given bounds
        in_bounds = (df.longitudes > lon[0]) & (df.longitudes < lon[1]) & \
                    (df.latitudes > lat[0]) & (df.latitudes < lat[1])
        lons = df.longitudes[in_bounds]
        lats = df.latitudes[in_bounds]
        aod_diff = df.data[1, in_bounds] - df.data[0, in_bounds]
        
        # If AERONET is included plot the sites on a map
        if np.any([df.data_sets[i] == 'aeronet' for i in [0,1]]):
            # Get the list of data points at each location
            site_lons, i_site = np.unique(df.longitudes, return_index=True)
            site_lats = df.latitudes[i_site]
            in_sites = site_lons[:, np.newaxis] == df.longitudes
            # Average the AOD at each site and take std
            aod_site_avg = np.mean(aod_diff * in_sites, axis=1)
#             aod_site_std = np.mean(aod_diff**2 * in_sites, axis=1) - aod_site_avg**2
            
            plt.scatter(site_lons, site_lats, c=aod_site_avg, cmap=match_frame_cmap, s=100)
            ax.coastlines()
            plt.colorbar(orientation='horizontal')
            plt.show()
            return
        
        # PLOT A GRID
        # Using scipy.interpolate.griddata
        # First get the axes
        grid = np.mgrid[(lon[0] + grid_size/2) : lon[1] : grid_size,
                        (lat[0] + grid_size/2) : lat[1] : grid_size]
        ll = zip(lons, lats)
        
        aod_grid = griddata(ll, aod_diff, tuple(grid), method='linear')
        
        # Mask grid data where there are no nearby points. Firstly create kd-tree
        THRESHOLD = 2 * grid_size   # Maximum distance to look for nearby points
        tree = cKDTree(ll)
        xi = _ndim_coords_from_arrays(tuple(grid))
        dists = tree.query(xi)[0]
        # Copy original result but mask missing values with NaNs
        aod_grid[dists > THRESHOLD] = np.nan
                  
        if plot_type == 'pcolormesh':
            plt.pcolormesh(grid[0], grid[1], aod_grid, cmap=match_frame_cmap)
        elif plot_type == 'contourf':
            plt.contourf(grid[0], grid[1], aod_grid, cmap=match_frame_cmap)
        
    
    ax.coastlines()
    plt.colorbar(orientation='horizontal')
    plt.show()