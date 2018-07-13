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
from matplotlib import pyplot as plt, cm
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
import matplotlib
sys.path.append('/home/h01/savis/workspace/summer')
import aeroct

# Suppress warnings from importing iris.plot in python 2
import warnings
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=MatplotlibDeprecationWarning)
    from iris import plot as iplt, analysis

# How to output the names of the data sets
name = {'aeronet': 'AERONET', 'modis': 'MODIS', 'metum': 'Unified Model'}


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero
    
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 1.0.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
      
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
        
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


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
    

def plot_map(df, lat=(-90,90), lon=(-180,180), plot_type='pcolormesh',
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
    
    # Latitude and longitude bounds
    in_bounds = (df.longitudes > lon[0]) & (df.longitudes < lon[1]) & \
                (df.latitudes > lat[0]) & (df.latitudes < lat[1])
    lons = df.longitudes[in_bounds]
    lats = df.latitudes[in_bounds]
    plt.xlim(lon)
    plt.ylim(lat)
    
    data_frame_cmap = cm.get_cmap('Oranges')
    match_frame_cmap = cm.get_cmap('RdBu_r')
    
    if df.__class__.__name__ == 'DataFrame':
        plt.title('Daily AOD for {} from {}'.format(df.date.date(), name[df.data_set]))
        
        if type(df.cube) == type(None):
            # Get data in bounds
            aod = df.aod[in_bounds]
        
        # USE IRIS PLOT IF THERE IS A CUBE IN THE DATA FRAME
        if type(df.cube) != type(None):
            
            day_avg_cube = df.cube.collapsed('time', analysis.MEAN)
            
            if plot_type == 'pcolormesh':
                iplt.pcolormesh(day_avg_cube, cmap=data_frame_cmap)
            if plot_type == 'contourf':
                iplt.contourf(day_avg_cube, cmap=data_frame_cmap)
        
        # PLOT AOD AT AERONET SITES AS A SCATTER PLOT
        elif df.data_set == 'aeronet':
            # Get the list of data points at each location
            site_lons, i_site = np.unique(lons, return_index=True)
            site_lats = lats[i_site]
            in_sites = site_lons[:, np.newaxis] == lons
            # Average the AOD at each site and take std
            aod_site_avg = np.mean(aod * in_sites, axis=1)
#             aod_site_std = np.sqrt(np.mean(aod**2 * in_sites, axis=1) - aod_site_avg**2)
            
            plt.scatter(site_lons, site_lats, c=aod_site_avg, cmap=data_frame_cmap, s=100)
        
        # PLOT THE AOD AT EVERY DATA POINT
        elif plot_type == 'scatter':
            plt.scatter(lons, lats, c=aod,
                        marker='o', s=(72./fig.dpi)**2)
        
        # OTHERWISE PLOT A GRID
        else:
            # Using scipy.interpolate.griddata
            # First get the axes
            grid = np.mgrid[(lon[0] + grid_size/2) : lon[1] : grid_size,
                                          (lat[0] + grid_size/2) : lat[1] : grid_size]
            
            ll = zip(lons, lats)
            aod_grid = griddata(ll, aod, tuple(grid), method='linear')
            
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
                  .format(df.date.date(), name[df.data_sets[1]], name[df.data_sets[0]]))
        
        # Find the data within the given bounds
        aod_diff = df.data_f[1, in_bounds] - df.data_f[0, in_bounds]
        data_min = np.min(aod_diff)
        data_max = np.max(aod_diff)
        cmap = shiftedColorMap(match_frame_cmap,
                               midpoint=data_min/(data_min-data_max))
        
        # USE IRIS PLOT IF THERE IS A CUBE IN THE DATA FRAME
        if type(df.cube) != type(None):
            
            day_avg_cube = df.cube.collapsed('time', analysis.MEAN)
            data_min = np.min(day_avg_cube.data)
            data_max = np.max(day_avg_cube.data)
            cmap = shiftedColorMap(match_frame_cmap, midpoint=data_min/(data_min-data_max))
            
            if plot_type == 'pcolormesh':
                iplt.pcolormesh(day_avg_cube, cmap=cmap)
            if plot_type == 'contourf':
                iplt.contourf(day_avg_cube, cmap=cmap)
        
        # If AERONET is included plot the sites on a map
        elif np.any([df.data_sets[i] == 'aeronet' for i in [0,1]]):
            # Get the list of data points at each location
            site_lons, i_site = np.unique(lons, return_index=True)
            site_lats = lats[i_site]
            in_sites = site_lons[:, np.newaxis] == lons
            # Average the AOD at each site and take std
            aod_site_avg = np.mean(aod_diff * in_sites, axis=1)
#             aod_site_std = np.sqrt(np.mean(aod_diff**2 * in_sites, axis=1) - aod_site_avg**2)
            
            # Shift colourmap to only include average site data
            data_min = np.min(aod_site_avg)
            data_max = np.max(aod_site_avg)
            cmap = shiftedColorMap(match_frame_cmap,
                                   midpoint=data_min/(data_min-data_max))
            
            plt.scatter(site_lons, site_lats, c=aod_site_avg, s=100,
                        cmap=cmap)
        
        # OTHERWISE PLOT A GRID
        else:
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
                plt.pcolormesh(grid[0], grid[1], aod_grid, cmap=cmap)
            elif plot_type == 'contourf':
                plt.contourf(grid[0], grid[1], aod_grid, cmap=cmap)
        
    
    ax.coastlines()
    plt.colorbar(orientation='horizontal')
    if show == True:
        plt.show()
        return
    else:
        return fig


def scatter_plot(df, stats=True, show=True, error=True, hm_threshold=500, **kwargs):
    '''
    This is used to plot AOD data from two sources which have been matched-up on a
    scatter plot. The function returns the figure if show=True.
    
    Parameters:
    df : AeroCT MatchFrame
        The data frame containing collocated data for a day.
    stats : bool, optional (Default: True)
        Choose whether to show statistics on the plot.    
    show : bool, optional (Default: True)
        If True, the plot is shown otherwise the figure is passed as an output.    
    error : bool, optional (Default: True)
        If True, error bars for the standard deviations are included on the plot.
    hm_threshold : int, optional (Default: 500)
        The threshold of number of data points above which a heat map will be plotted
        instead of a scatter plot.
    **kwargs : optional
        Arguments for the style of the scatter plot. By default c='r', marker='o',
        linestyle='None' and, if error=True, ecolor='gray'.
    '''
    if (df.__class__.__name__ != 'MatchFrame') | (len(df.data_sets) != 2):
        raise ValueError('The data frame is unrecognised. It must be collocated data \
                         from two data sets.')
    
    fig, ax = plt.subplots()
    
    # Plot a scatter plot if there are fewer data points than hm_threshold
    if (df.data_f[0].size <= hm_threshold):
        kwargs.setdefault('c', 'r')
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        
        if error == True:
            kwargs.setdefault('ecolor', 'gray')
            plt.errorbar(df.data_f[0], df.data_f[1], df.std_f[1], df.std_f[0], **kwargs)
        else:
            plt.plot(df.data_f[0], df.data_f[1], **kwargs)
    
    # Otherwise plot a heatmap
    else:
        x_min, x_max = np.min(df.data_f[0]), np.max(df.data_f[0])
        y_min, y_max = np.min(df.data_f[1]), np.max(df.data_f[1])
        x_grid = np.linspace(x_min, x_max, 101)
        y_grid = np.linspace(y_min, y_max, 101)
        
        # Find the number of points in each grid cell and mask those with none
        heatmap_grid = np.histogram2d(df.data_f[0], df.data_f[1], [x_grid, y_grid])[0]
        heatmap_grid = np.ma.masked_where(heatmap_grid==0, heatmap_grid)
        
        plt.pcolormesh(x_grid, y_grid, heatmap_grid.T, cmap='CMRmap')
        plt.colorbar(orientation='horizontal')
    
    ax.autoscale(False)
    
    # Regression line
    x = np.array([0, 10])
    y = df.R_INTERCEPT + x * df.R_SLOPE
    plt.plot(x, y, 'g:', lw=2, label='Regression')
    
    # y = x line
    plt.plot([0, 10], [0, 10], c='gray', ls='--', lw=2, label='y = x')
    
    # Title, axes, and legend 
    plt.title('Collocated AOD comparison on {0}'.format(df.date.date()))
    plt.legend(loc=4)
    if df.forecast_times[0] != None:
        plt.xlabel('{} (forecast lead time: {} hours)'\
                   .format(name[df.data_sets[0]], int(df.forecast_times[0])))
    else:
        plt.xlabel(name[df.data_sets[0]])
    if df.forecast_times[1] != None:
        plt.ylabel('{} (forecast lead time: {} hours)'\
                   .format(name[df.data_sets[1]], int(df.forecast_times[1])))
    else:
        plt.ylabel(name[df.data_sets[1]])
    
    # Stats
    if stats == True:
        rms_str = 'RMS: {:.02f}'.format(df.RMS)
        plt.text(0.03, 0.94, rms_str, fontsize=12, transform=ax.transAxes)
        bias_mean_str = 'Bias mean: {:.02f}'.format(df.BIAS_MEAN)
        plt.text(0.03, 0.88, bias_mean_str, fontsize=12, transform=ax.transAxes)
        bias_std_str = 'Bias std: {:.02f}'.format(df.BIAS_STD)
        plt.text(0.03, 0.82, bias_std_str, fontsize=12, transform=ax.transAxes)
        r_str = 'Pearson R: {:.02f}'.format(df.R)
        plt.text(0.35, 0.94, r_str, fontsize=12, transform=ax.transAxes)
        slope_str = 'Slope: {:.02f}'.format(df.R_SLOPE)
        plt.text(0.35, 0.88, slope_str, fontsize=12, transform=ax.transAxes)
        intercept_str = 'Intercept: {:.02f}'.format(df.R_INTERCEPT)
        plt.text(0.35, 0.82, intercept_str, fontsize=12, transform=ax.transAxes)
    
    if show == True:
        plt.show()
        return
    else:
        return fig