'''
This module contains the functions to plot single DataFrames or MatchFrames, ie. a single
day.
The AOD data throughout the day for a single AERONET site can be plotted with
plot_anet_site().
The daily average for the AOD / AOD difference for either a DataFrame or MatchFrame can
be plotted on a map using plot_map().
The AOD match-up for a MatchFrame may be plotted on a scatter plot with scatterplot().

Created on Jul 5, 2018

@author: savis
'''
from __future__ import division
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib import pyplot as plt, cm, animation, widgets, patches, dates as mdates
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from scipy.stats import linregress

import aeroct

# Suppress warnings from importing iris.plot in python 2
import warnings
from __builtin__ import isinstance
try:
    from matplotlib.cbook.deprecation import mplDeprecation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=mplDeprecation)
        from iris import plot as iplt, analysis
except:
    from iris import plot as iplt, analysis


def shiftedColorMap(cmap, data, vmin=None, vmax=None, midpoint=0, name='shiftedcmap'):
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
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    
    
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    if (vmin == None) & (vmin == None):
        idx_mid = (midpoint - data_min) / (data_max - data_min)
    elif vmin == None:
        idx_mid = (midpoint - data_min) / (vmax - data_min)
    elif vmax == None:
        idx_mid = (midpoint - vmin) / (data_max - vmin)
    else:
        idx_mid = (midpoint - vmin) / (vmax - vmin)
    
    # Ensure the midpoint is within 0-1
    if idx_mid < 0:
        idx_mid = 0
    elif idx_mid > 1:
        idx_mid = 1
    
    # regular index to compute the colors
    reg_index = np.linspace(0, 1, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, idx_mid, 128, endpoint=False), 
        np.linspace(idx_mid, 1.0, 129, endpoint=True)
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


def plot_map(df, data_type='AOD', lat=(-90,90), lon=(-180,180), plot_type='pcolormesh',
             show=True, grid_size=0.5, vmin=None, vmax=None):
    '''
    For DataFrames this function will plot the daily average of the AOD at individual
    sites for AERONET data, otherwise on a grid.
    For MatchFrames this the difference in AOD is plotted (data_set[1] - data_set[0]).
    This will be displayed as individual sites if AERONET data is included, otherwise on
    a grid.
    
    Parameters
    ----------
    df : AeroCT DataFrame / MatchFrame, or list of DataFrames / MatchFrames
    data_type : str, optional (Default: 'AOD')
        This describes which type of data to plot.
        If df is a DataFrame(s) then it describes the type of AOD data to plot. It can
        take the values 'AOD', 'dust AOD', or 'total AOD'. If 'AOD' is chosen then the
        total AOD will be plotted, unless the data frame has only dust AOD in which case
        that will be plotted.
        If df is a MatchFrame(s) then it can take the 
    lat : tuple, optional (Default: (-90, 90))
        A tuple of the latitude bounds of the plot in degrees.
    lon : tuple, optional (Default: (-180, 180))
        A tuple of the longitude bounds of the plot in degrees.
    plot_type : str, optional (Default: 'pcolormesh')
        The type of plot to produce if it does not contain AERONET data. 'scatter' to
        plot a scatter grid of all AOD data, and 'pcolormesh' or 'contourf' for griddded
        plots.
    show : bool, optional (Default: True)
        If True the figure is shown, otherwise it is returned 
    grid_size : float, optional (Default: 0.5)
        The size of the grid squares in degrees if not using indivdual sites
    vmin, vmax : scalar, optional (Default: None)
        vmin and vmax are used to set limits for the color map. If either is None, it is
        autoscaled to the respective min or max of the plotted data.
    '''
    
    # Convert a list of match frames to a single match frame that may be plotted
    if isinstance(df, list):
        df = aeroct.concatenate_data_frames(df)
        date = '{0} to {1}'.format(df.date[0].date(), df.date[-1].date())
    elif isinstance(df.date, list):
        date = '{0} to {1}'.format(df.date[0].date(), df.date[-1].date())
    else:
        date = df.date.date()
    
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    plt.xlim(lon)
    plt.ylim(lat)
    
    mon_cmap = cm.get_cmap('inferno_r')
    div_cmap = cm.get_cmap('RdYlBu_r')
    
    # USE IRIS PLOT IF THERE IS A CUBE IN THE DATA FRAME
    if type(df.cube) != type(None):
        
        day_avg_cube = df.cube.collapsed('time', analysis.MEAN)
        
        if df.__class__.__name__ == 'DataFrame':
            plt.title('{0}: AOD mean for {1}'.format(df.name, date))
            cmap = mon_cmap
        else:
            plt.title('AOD difference (mean) : {0} - {1} for {2}'\
                      .format(df.names[1], df.names[0], date))
            cmap = shiftedColorMap(div_cmap, day_avg_cube.data, vmin, vmax)
        
        if plot_type == 'pcolormesh':
            iplt.pcolormesh(day_avg_cube, cmap=cmap, vmin=vmin, vmax=vmax)
        elif plot_type == 'contourf':
            iplt.contourf(day_avg_cube, cmap=cmap, vmin=vmin, vmax=vmax)
    
    elif df.__class__.__name__ == 'DataFrame':
        plt.title('{0}: AOD Mean For {1}'.format(df.name, date))
        cmap = mon_cmap
        
        # Select the total or dust AOD data which is in the given bounds
        if data_type == 'AOD':
            aod, lons, lats = df.get_data(None)[:3]
        if data_type in ['total', 'total AOD']:
            aod, lons, lats = df.get_data('total')[:3]
        if data_type in ['dust', 'dust AOD']:
            aod, lons, lats = df.get_data('dust')[:3]
        
        in_bounds = (lon[0] < lons) & (lons < lon[1]) & (lat[0] < lats) & (lats < lat[1])
        aod = aod[in_bounds]
        lons = lons[in_bounds]
        lats = lats[in_bounds]
        
        # PLOT AOD AT AERONET SITES AS A SCATTER PLOT
        if df.data_set == 'aeronet':
            # Get the list of data points at each location
            site_lons, i_site = np.unique(lons, return_index=True)
            site_lats = lats[i_site]
            in_sites = site_lons[:, np.newaxis] == lons
            # Average the AOD at each site and take std
            aod_site_avg = np.mean(aod * in_sites, axis=1)
#             aod_site_std = np.sqrt(np.mean(aod**2 * in_sites, axis=1) - aod_site_avg**2)
            
            plt.scatter(site_lons, site_lats, c=aod_site_avg, cmap=cmap,
                        edgecolors='k', linewidths=0.2, s=100, vmin=vmin, vmax=vmax)
        
        # PLOT THE AOD AT EVERY DATA POINT
        elif plot_type == 'scatter':
            plt.scatter(lons, lats, c=aod, marker='o', s=(50./fig.dpi)**2,
                        vmin=vmin, vmax=vmax)
        
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
                plt.pcolormesh(grid[0], grid[1], aod_grid, cmap=cmap,
                               vmin=vmin, vmax=vmax)
            elif plot_type == 'contourf':
                plt.contourf(grid[0], grid[1], aod_grid, cmap=cmap,
                             vmin=vmin, vmax=vmax)
    
    elif df.__class__.__name__ == 'MatchFrame':        
        # Find the data within the given longitude and latitude bounds
        in_bounds = (df.longitudes > lon[0]) & (df.longitudes < lon[1]) & \
                    (df.latitudes > lat[0]) & (df.latitudes < lat[1])
        lons = df.longitudes[in_bounds]
        lats = df.latitudes[in_bounds]
        
        # Get the data for which to find the area average and the title
        if data_type == 'time diff':
            data = df.time_diff[in_bounds]
            plt.title('Time Difference (mean) ({0} - {1}) for {2}'\
                      .format(df.names[1], df.names[0], date))
        
        elif data_type == 'RMS':
            data = (df.data[1, in_bounds] - df.data[0, in_bounds])**2
            plt.title('Root Mean Square ({0}, {1}) for {2}'\
                      .format(df.names[0], df.names[1], date))
        
        elif data_type == 'heatmap':
            data = np.ones_like(df.data[1, in_bounds])
            plt.title('Number Of Matched Data Points ({0}, {1}) for {2}'\
                      .format(df.names[0], df.names[1], date))
        
        else:
            data = df.data[1, in_bounds] - df.data[0, in_bounds]
            plt.title('AOD Difference (mean) ({0} - {1}) for {2}'\
                      .format(df.names[1], df.names[0], date))
        
        # If AERONET is included plot the sites on a map
        if np.any([df.data_sets[i] == 'aeronet' for i in [0,1]]):
            # Get the list of data points at each location
            site_lons, i_site = np.unique(lons, return_index=True)
            site_lats = lats[i_site]
            in_sites = site_lons[:, np.newaxis] == lons
            # Average the AOD at each site and take std
            site_data = np.mean(data * in_sites, axis=1)
            
            if data_type == 'RMS':
                site_data = np.sqrt(site_data)
                cmap = mon_cmap
            elif data_type == 'heatmap':
                site_data = np.sum(in_sites, axis=1)
                cmap = mon_cmap
            else:
                # Shift colour map to have a midpoint of zero
                cmap = shiftedColorMap(div_cmap, site_data, vmin, vmax)
            
            plt.scatter(site_lons, site_lats, c=site_data, s=(500/fig.dpi)**2,
                        edgecolors='k', linewidths=0.2, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # OTHERWISE PLOT A GRID
        else:
            # Using scipy.interpolate.griddata
            # First get the axes
            grid = np.mgrid[(lon[0] + grid_size/2) : lon[1] : grid_size,
                            (lat[0] + grid_size/2) : lat[1] : grid_size]
            ll = zip(lons, lats)
            
            if data_type == 'heatmap':
                lon_edges = np.arange(lon[0], lon[1]+1e-5, grid_size)
                lat_edges = np.arange(lat[0], lat[1]+1e-5, grid_size)
                data_grid = np.histogram2d(lons, lats, [lon_edges, lat_edges])[0]
            else:
                data_grid = griddata(ll, data, tuple(grid), method='cubic')
            
            # Mask grid data where there are no nearby points. Firstly create kd-tree
            THRESHOLD = grid_size   # Maximum distance to look for nearby points
            tree = cKDTree(ll)
            xi = _ndim_coords_from_arrays(tuple(grid))
            dists = tree.query(xi)[0]
            # Copy original result but mask missing values with NaNs
            data_grid[dists > THRESHOLD] = np.nan
            
            if data_type == 'RMS':
                data_grid = np.sqrt(data_grid)
            
            if data_type in ['RMS', 'heatmap']:
                cmap = mon_cmap
            else:
                # Shift colour map to have a midpoint of zero
                cmap = shiftedColorMap(div_cmap, data_grid, vmin, vmax)
            
            if plot_type == 'pcolormesh':
                plt.pcolormesh(grid[0], grid[1], data_grid, cmap=cmap,
                               vmin=vmin, vmax=vmax)
            elif plot_type == 'contourf':
                plt.contourf(grid[0], grid[1], data_grid, cmap=cmap, vmin=vmin, vmax=vmax)
        
    
    ax.coastlines()
    plt.colorbar(orientation='horizontal')
    
    fig.tight_layout()
    if show == True:
        plt.show()
        return
    else:
        return fig


def scatterplot(mf, aeronet_site=None, stats=True, scale='log', xlim=(None, None),
                ylim=(None, None), show=True, error=True, hm_threshold=400,
                grid_cells=None, **kwargs):
    '''
    This is used to plot AOD data from two sources which have been matched-up on a
    scatter plot. The data for a single AERONET site may be shown if one of the matched
    data-sets is AERONET. The function returns the figure if show=True.
    
    Parameters
    ----------
    mf : AeroCT MatchFrame or list of MatchFrames
        The data frame(s) containing collocated data.
    aeronet_site : str, optional (Default: None)
        The name of the AERONET site for which to show data If a single site is desired.
    stats : bool, optional (Default: True)
        Choose whether to show statistics on the plot.
    scale : {'log', 'linear', 'bins'}, optional (Default: 'log')
        Choose whether to plot the data on a log scale (if so anything below 1e-4 is not
        displayed).
    xlim, ylim : float tuple (Defaults: (None, None))
        Limits for the plot. By default it autoscales to the data.
    show : bool, optional (Default: True)
        If True, the plot is shown otherwise the figure is passed as an output.    
    error : bool, optional (Default: True)
        If True, error bars for the standard deviations are included on the plot.
    hm_threshold : int, optional (Default: 400)
        The threshold of number of data points above which a heat map will be plotted
        instead of a scatter plot.
    grid_cells : int, optional (Default: None)
        This changes the number of grid cellsalong an axis for a heatmap and the
        histograms. If None this is calculated by: 20 * (num of matches)**0.2
    **kwargs : optional
        Arguments for the style of the scatter plot. By default c='r', marker='o',
        linestyle='None' and, if error=True, ecolor='gray'.
    '''
    if isinstance(mf, list):
        mf = aeroct.concatenate_data_frames(mf)
        date_str = '{0} to {1}'.format(mf.date[0].date(), mf.date[-1].date())
    elif isinstance(mf.date, list):
        date_str = '{0} to {1}'.format(mf.date[0].date(), mf.date[-1].date())
    else:
        date_str = '{0}'.format(mf.date.date())
    
    if (mf.__class__.__name__ != 'MatchFrame') | (len(mf.data_sets) != 2):
        raise ValueError('The data frame is unrecognised. It must be collocated data \
                         from two data sets.')
    
    # Select the data from the given AERONET site
    if aeronet_site is not None:
        mf = mf.extract(aeronet_site=aeronet_site)
    
    # Plot a heat map if there are more data points than hm_threshold
    heatmap = (mf.data[0].size > hm_threshold)
    
    fig = plt.figure(figsize=(8,8))
    
    # Axes locations and sizes
    x0, y0 = 0.13, 0.08
    width, height = 0.7, 0.6
    width2, height2 = 0.1, 0.1
    cheight, cpad = 0.03, 0.1
    
    if heatmap:
        cax = fig.add_axes([x0, y0, width, cheight])
        y1 = y0 + cheight + cpad
    else:
        y1 = y0 + (cheight + cpad) / 2
    
    ax = fig.add_axes([x0, y1, width, height])
    ax_x = fig.add_axes([x0, y1 + height + 0.01, width, height2], sharex=ax)
    ax_y = fig.add_axes([x0 + width + 0.01, y1, width2, height], sharey=ax)
    
    # Grid cell boundaries for heat-map / histograms
    if xlim[0] is None:
        xmax = np.max(mf.data[0])
        data_min = np.min(mf.data[0, mf.data[0] > 1e-6])
        xmin = 1e-3 if (data_min < 1e-3) else data_min
    else:
        xmax = xlim[1]
        xmin = 1e-3 if (xlim[0] < 1e-3) else xlim[0]
    
    if ylim[0] is None:
        ymax = np.max(mf.data[1])
        data_min = np.min(mf.data[1, mf.data[1] > 1e-6])
        ymin = 1e-3 if (data_min < 1e-3) else data_min
    else:
        ymax = ylim[1]
        ymin = 1e-3 if (ylim[0] < 1e-3) else ylim[0]
    
    # Grid cells
    if grid_cells is None:
        grid_cells = 20 * mf.num ** 0.2
    grid_cells += 1
    if (scale == 'log') | ((not heatmap) & (scale=='bins')):
        x_grid = 10 ** np.linspace(np.log10(xmin), np.log10(xmax), grid_cells)
        y_grid = 10 ** np.linspace(np.log10(ymin), np.log10(ymax), grid_cells)
    elif scale == 'linear':
        x_grid = np.linspace(xmin, xmax, grid_cells)
        y_grid = np.linspace(ymin, ymax, grid_cells)
    elif scale == 'bins':
        bins = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
        x_grid = np.array(bins)
        y_grid = np.array(bins)
    
    # Plot a scatter plot if there are fewer data points than hm_threshold
    if not heatmap:
        kwargs.setdefault('c', 'r')
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        
        if error == True:
            kwargs.setdefault('ecolor', 'gray')
            ax.errorbar(mf.data[0], mf.data[1], mf.data_std[1], mf.data_std[0], zorder=0, **kwargs)
        else:
            ax.plot(mf.data[0], mf.data[1], zorder=0, **kwargs)
    
    # Otherwise plot a heat-map
    else:
        # Find the number of points in each grid cell and mask those with none
        heatmap_grid = np.histogram2d(mf.data[0], mf.data[1], [x_grid, y_grid])[0]
        heatmap_grid = np.ma.masked_where(heatmap_grid==0, heatmap_grid)
        
        im = ax.pcolormesh(x_grid, y_grid, heatmap_grid.T, cmap='CMRmap_r')
        plt.colorbar(im, cax=cax, orientation='horizontal')
    
    # Assign automatic limits now so that they do not fit to the lines
    if xlim[0] is None:
        xlim = ax.get_xlim()
    if ylim[0] is None:
        ylim = ax.get_ylim()
    
    x = np.linspace(1e-4, 10, 1001)
    y = mf.r_intercept + x * mf.r_slope
    x_data = mf.data[0][(mf.data[0] > 0) & (mf.data[1] > 0)]
    y_data = mf.data[1][(mf.data[0] > 0) & (mf.data[1] > 0)]
    log_r_slope, log_r_intercept, log_r = \
                linregress(np.log10(x_data), np.log10(y_data))[:3]
    y_log = 10 ** (log_r_intercept + np.log10(x) * log_r_slope)
    
#     ax.plot(x, y, 'g-.', lw=2, label='Linear regression') # Regression line
#     ax.plot(x, y_log, 'b-', lw=2, label='Logarithmic regression') # Regression line
    ax.plot(x, x, c='gray', ls='--', lw=2, label='y = x') # y = x line
    
    # AOD source for dust
    if mf.aod_type == 'dust':
        aod_src = {'metum' : '',
                   'modis' : '(Filtered coarse AOD)',
                   'modis_t' : '(Filtered coarse AOD)',
                   'modis_a' : '(Filtered coarse AOD)',
                   'aeronet' : '(SDA coarse mode)'}
    else:
        aod_src = {'metum' : '', 'modis': '', 'modis_t' : '', 'modis_a' : '', 'aeronet': ''}
    
    # Title, axes, and legend
    if aeronet_site is not None:
        rgn_str = aeronet_site
    elif np.any([mf.additional_data[i][:9]=='Extracted'
                 for i in range(len(mf.additional_data))]):
        rgn_str = 'Regional'
    else:
        rgn_str = 'Global'
    
    if mf.aod_type in ('total', 0):
        aod_str = 'Total AOD'
    elif mf.aod_type in ('dust', 1):
        aod_str = 'Dust AOD'
    
    title = 'Collocated {1} Comparison \nFor {2} ({0})'.format(rgn_str, aod_str, date_str)
    
    fig.text(0.5, (y1 + height + height2 + 0.03), title, ha='center', fontsize=14)
    ax.legend(loc=4)
    ax.set_xlabel('{0} AOD {1}'.format(mf.names[0], aod_src[mf.data_sets[0]]))
    ax.set_ylabel('{0} AOD {1}'.format(mf.names[1], aod_src[mf.data_sets[1]]))
    if heatmap:
        cax.set_xlabel('Match-ups in each cell')
    
    if scale=='log':
        ax.loglog()
        xlim = (1e-3, xlim[1]) if (xlim[0] <= 1e-3) else xlim
        ylim = (1e-3, ylim[1]) if (ylim[0] <= 1e-3) else ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Ticks
    ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax_x.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
    ax_y.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True, labelleft=False)
    if heatmap & (scale=='bins'):
        ax.loglog()
    
    # Histograms
    ax_x.hist(mf.data[0], bins=x_grid, color='k')
    ax_y.hist(mf.data[1], bins=y_grid, color='k', orientation='horizontal')
    
    # Stats
    if stats == True:
        box = dict(facecolor='w', edgecolor='w', pad=-0.75)
        num_str = 'Num: {:d}'.format(mf.num)
        plt.text(0.03, 0.94, num_str, fontsize=12, transform=ax.transAxes, bbox=box)
        
        rms_str = 'RMS: {:.02f}'.format(mf.rms)
        plt.text(0.03, 0.88, rms_str, fontsize=12, transform=ax.transAxes, bbox=box)
        
        bias_mean_str = 'Bias mean: {:.03f}'.format(mf.bias_mean)
        plt.text(0.03, 0.82, bias_mean_str, fontsize=12, transform=ax.transAxes, bbox=box)
        
        bias_std_str = 'Bias std: {:.03f}'.format(mf.bias_std)
        plt.text(0.03, 0.76, bias_std_str, fontsize=12, transform=ax.transAxes, bbox=box)
        
        r_str = 'Pearson R: {:.02f}'.format(mf.r)
        plt.text(0.4, 0.94, r_str, fontsize=12, transform=ax.transAxes, bbox=box)
        
        slope_str = 'Slope: {:.02f}'.format(mf.r_slope)
        plt.text(0.4, 0.88, slope_str, fontsize=12, transform=ax.transAxes, bbox=box)
        
        intercept_str = 'Intercept: {:.02f}'.format(mf.r_intercept)
        plt.text(0.4, 0.82, intercept_str, fontsize=12, transform=ax.transAxes, bbox=box)
    
    fig.tight_layout()
    if show == True:
        plt.show()
        return
    else:
        return fig


def plot_time_series(mf_lists, stat, xlim=None, ylim=None, average_days=None, show=True):
    '''
    Given a list containing MatchFrames a certain daily statistic is plotted over time.
    Additionally multiple of these lists may be passed in which case they will be
    plotted on the same axes.
    
    Parameters
    ----------
    mf_lists : list of MatchFrames (or list of MatchFrame lists)
        A list may be obtained using the get_match_list() function. If a list of lists is
        supplied then the individual lists will be plotted on the same axes. Note that
        these lists should all have the same 'data set 1' or the plots title will be
        inaccurate.
    stat : str
        This gives the type of statistic to plot over time. Options are:
        'RMS' - Root mean square error
        'RMS norm' - Root mean square error normalised by the mean AOD
        'Bias' - AOD bias mean (data set 2 - data set 1)
        'Bias norm' - AOD bias mean normalised by the mean AOD
        'R'- Pearson correlation coefficient to the regression line
        'Number' - Number of daily match-ups
    xlim : datetime tuple, optional (Default: None)
        The limits for the date. If None then the axis is autoscaled to the data.
    ylim : float tuple, optional (Default: None)
        The limits for the AOD bias. If None then the axis is autoscaled to the data.
    average_days : int, optional (Default: None)
        The number of days over which to perform a running average. If None then the
        total number of days is divided by 50 and rounded.
    show : bool, optional (Default: True)
        If True, the plot is shown otherwise the figure is passed as an output.    
    '''
    if not isinstance(mf_lists[0], list):
        mf_lists = [mf_lists]
    
    if average_days == None:
        average_days = int(len(mf_lists[0]) / 50)
        average_days = 1 if (average_days == 0) else average_days
    
    fig, ax = plt.subplots()
    
    for mf_list in mf_lists:
        date_list = [mf.date for mf in mf_list]
        
        # Get the values of the statistic for every day
        if stat == 'RMS':
            stat_name = 'Root Mean Square'
            stat_values = [mf.rms for mf in mf_list]
        elif stat == 'RMS norm':
            stat_name = 'Normalised Root Mean Square'
            stat_values = [mf.rms / np.mean(mf.data**2)**.5 for mf in mf_list]
        elif stat == 'Bias':
            stat_name = 'Mean Bias (data set 2 - data set 1)'
            stat_values = [mf.bias_mean for mf in mf_list]
            stat_errors = [mf.bias_std for mf in mf_list]
        elif stat == 'Bias norm':
            stat_name = 'Normalised Mean Bias (data set 2 - data set 1)'
            stat_values = [mf.bias_mean / np.mean(mf.data) for mf in mf_list]
            stat_errors = [mf.bias_std / np.mean(mf.data) for mf in mf_list]
        elif stat == 'R':
            stat_name = 'Pearson Correlation Coefficient'
            stat_values = [mf.r for mf in mf_list]
        elif stat == 'Number':
            stat_name = 'Number of Matched Points'
            stat_values = [mf.num for mf in mf_list]
        else:
            raise ValueError('Invalid statistic: {0}'.format(stat))
                
        # Average over a number of days
        date_list_reduced = date_list[int((average_days - 1)/2)::average_days]
        if average_days > 2: date_list_reduced.append(date_list[-1])
        stat_mean, stat_err = aeroct.average_each_n_values(stat_values, average_days)
        
        if stat == 'Bias':
            stat_err = aeroct.average_each_n_values(stat_errors, average_days)[0]
        
        # Plot
        if (len(mf_lists) == 1):
            plt.errorbar(date_list_reduced, stat_mean, stat_err, ecolor='gray', elinewidth=0.5)
        else:
            plt.plot(date_list_reduced, stat_mean, label=mf.names[1])
    
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    plt.hlines(0, datetime(1970,1,1), datetime(2100,1,1), linestyle='--', lw=0.5, zorder=-1)
    
    # Title and axes
    plt.xlabel('Date')
    plt.ylabel(stat_name)
    ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
    title1 = 'Daily statistic Over Time For Collocated {0} AOD data'.format(mf.aod_type)
    if len(mf_lists) > 1:
        title2 = '\nData Set 1: {0}, Data Set 2: See Legend'.format(mf.names[0])
    else:
        title2 = '\nData Set 1: {0}, Data Set 2: {1}'.format(mf.names[0], mf.names[1])
    plt.title(title1 + title2)
    
    # Format date axis
    fig.autofmt_xdate()
    dates_fmt = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(dates_fmt)
    
    if len(mf_lists) > 1:
        plt.legend(loc='best')
    
    fig.tight_layout()
    if show == True:
        plt.show()
    else:
        return fig


def grid_mean(data, lons, lats, grid):
    '''
    Gets the mean value of data for each grid cell. Used in plot_map for RMS.
    '''
    # Regrid data
    grid_size = grid[1,0,1] - grid[1,0,0]
    lons = np.rint((lons.ravel() - grid_size/2) / grid_size) * grid_size + grid_size/2
    lats = np.rint((lats.ravel() - grid_size/2) / grid_size) * grid_size + grid_size/2
    
    # Sort the latitudes and longitudes to bring duplicate locations together
    ll = np.array([lons, lats])
    sort_idx = np.lexsort(ll)
    sorted_ll = ll[:, sort_idx]
    sorted_data = data.ravel()[sort_idx]
    
    # Obtain a mask to indicate where the new locations begin
    uniq_mask = np.append(True, np.any(np.diff(sorted_ll, axis=1), axis=0))
     
    # Find an ID number to identify the location of each member of the sorted arrays
    ID = np.full_like(sorted_data, np.nan, dtype=int)
    for i, new_loc in enumerate(uniq_mask):
        if new_loc:
            ID[i] = np.argmin(np.abs(sorted_ll[0,i]-grid[0].ravel() + sorted_ll[1,i]-grid[1].ravel()))
        else:
            ID[i] = ID[i-1]
    
    # Append the final index to ID so that bincount counts all gridcells
    ID = np.append(ID, grid[0].size - 1)
    sorted_data = np.append(sorted_data, np.nan)
     
    # Take the average and standard deviations for each location
    div0 = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
    mean_grid_data = div0(np.bincount(ID, sorted_data), np.bincount(ID))
    return mean_grid_data.reshape(grid[0].shape)


def period_bias_plot(mf_lists, xlim=None, ylim=None, show=True, **kw):
    '''
    Given a list containing MatchFrames the bias between the two sets of collocated AOD
    values are calculated. The mean bias for each day is plotted with an error bar
    containing the standard deviation of the bias.
    
    Parameters
    ----------
    mf_lists : list of MatchFrames (or list of MatchFrame lists)
        May be obtained using the get_match_list() function. The bias is the second data
        set AOD subtract the first. If a list of lists then the individual lists will be
        plotted on separate sub-plots.
    xlim : datetime tuple, optional (Default: None)
        The limits for the date. If None then the axis is autoscaled to the data.
    ylim : float tuple, optional (Default: None)
        The limits for the AOD bias. If None then the axis is autoscaled to the data.
    show : bool, optional (Default: True)
        Choose whether to show the plot. If False the figure is returned by the function.
    kwargs : optional
        These kwargs are passed to matplotlib.pyplot.errorbar() to format the plot. If
        none are supplied then the following are used:
        fmt='b.', markersize=2, ecolor='gray', capsize=0.
    '''
    if not isinstance(mf_lists[0], list):
        mf_lists = [mf_lists]
    
    subplots = len(mf_lists)
    fig, ax = plt.subplots(subplots, sharex=True, sharey=True, squeeze=False)
    ax = ax[:, 0]
    
    for i, mf_list in enumerate(mf_lists):
        plt.sca(ax[i])
        
        bias_arrays = [mf.data[1] - mf.data[0] for mf in mf_list]
        bias_mean = np.array([np.mean(bias_array) for bias_array in bias_arrays])
        bias_std = np.array([np.std(bias_array) for bias_array in bias_arrays])
        date_list = [mf.date for mf in mf_list]
        
        # Plot formatting
        kw.setdefault('fmt', 'bs')
        kw.setdefault('markersize', 2)
        kw.setdefault('ecolor', 'gray')
        kw.setdefault('elinewidth', 0.5)
        kw.setdefault('capsize', 0)
        
        plt.errorbar(date_list, bias_mean, bias_std, **kw)
        
        if xlim is None:
            ax[i].set_xlim(ax[i].get_xlim())
        else:
            ax[i].set_xlim(xlim)
        if ylim is None:
            ax[i].set_ylim(ax[i].get_ylim())
        else:
            ax[i].set_ylim(ylim)
        
        plt.hlines(0, datetime(1970,1,1), datetime(2100,1,1), linestyle='--', lw=0.5)
        
        ax[i].tick_params(direction='in', bottom=True, top=True, left=True, right=True)
        ax[i].tick_params(direction='in', bottom=True, top=True, left=True, right=True)
        if mf_list[0].data_sets[1] == 'metum':
            ax[i].set_ylabel('Lead Time: {0} hours'.format(int(mf_list[0].forecast_times[1])))
        elif mf_list[0].data_sets[1][:5] == 'modis':
            ax[i].set_ylabel('({0} - {1})'.format(mf_list[0].names[1], mf_list[0].names[0]))
    
    if mf_list[0].aod_type == 0: mf_list[0].aod_type = 'total'
    if mf_list[0].aod_type == 1: mf_list[0].aod_type = 'dust'
    plt.suptitle('{0} AOD Daily Mean Bias'.format(mf_list[0].aod_type.title()))
    plt.xlabel('Date')
    if mf_list[0].data_sets[1] == 'metum':
        ylabel = 'AOD bias (Unified Model - {1})'.format(mf_list[0].data_sets[1], mf_list[0].names[0])
    elif mf_list[0].data_sets[1][:5] == 'modis':
        ylabel = 'AOD bias'
    fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical')
    
    if show == True:
        plt.show()
    else:
        return fig


def plot_anet_site(mf, site, data_frames=None, aod_type='total'):
    '''
    Plot the daily AOD data for a single AERONET site.
    
    Parameters
    ----------
    aeronet : aeroct DataFrame
        A data-frame returned by aeroct.load() containing AERONET data for which to plot. 
    data_frames : list of aeroct DataFrames
        A list of other data-frames returned by aeroct.load() for which to plot.
    site : str
        The name of the AERONET site at which to plot data.
    aod_type : {'total' or 'dust'}, optional (Default: 'total')
        The type of AOD data to plot.
    '''
#     sites, i_uniq, i_inv = np.unique(aeronet.sites, return_index=True , return_inverse=True)
#     lat = aeronet.latitudes[i_uniq[sites==site]]
#     lon = aeronet.longitudes[i_uniq[sites==site]]
#     site_ll = np.array([lon, lat])
#     
#     # Get the AOD and times for the AERONET data at the chosen site
#     anet_times = aeronet.times[i_uniq[sites==site]]
#     if aod_type == 'total':
#         anet_aod = aeronet.aod[0][i_uniq[sites==site]]
#     elif aod_type == 'dust':
#         anet_aod = aeronet.aod[1][i_uniq[sites==site]]
#     
#     # Plot the AERONET data
#     plt.plot(anet_times, anet_aod, label='AERONET')
#     
#     for df in data_frames:
#         aod, times = aeroct.match_to_site(df, site_ll, aod_type, match_dist=25)
#         plt.plot(times, aod, label=df.name)
    
    lons, i_uniq, i_inv = np.unique(mf.longitudes, return_index=True , return_inverse=True)
    lon = lons[site]
    lat = mf.latitudes[i_uniq[site]]
    times = mf.times[i_inv == site]
    aod1 = mf.data[0, i_inv == site]
    aod2 = mf.data[1, i_inv == site]
     
    plt.plot(times, aod1, 'r.')
    plt.plot(times, aod2, 'c.')
    plt.title('Daily {0} AOD From AERONET Site: {1}'.format(mf.names[0].title(), site))
    if aod_type == 'both':
        plt.legend(loc='best')
    plt.show()


def plot_aod_hist(df, aod_type=None):
    if df.data_set == 'metum':
        if (aod_type is None) | (aod_type == 'dust'):
            aod = df.aod[1]
    else:
        aod = df.get_data(aod_type)[0]
    
    plt.hist(aod, bins=20)
    plt.show()
    

def plot_region_mask(bounds):
    '''
    Used to plot regions on a map which are bounded by longitude and latitude.
    
    Parameters
    ----------
    bounds : list of 4-tuples
        Each 4-tuple in this list corresponds to a region that will be plotted on the
        map. The 4-tuples contain the bounds as follows:
        (min lon, max lon, min lat, max lat)
    '''
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()
    ax.gridlines(linestyle='--', linewidth=0.3, draw_labels=True)
    
    if not isinstance(bounds[0], list):
        bounds = [bounds]
    
    for bound in bounds:
        width = bound[1] - bound[0]
        height = bound[3] - bound[2]
        region = patches.Rectangle((bound[0], bound[2]), width, height, linewidth=2,
                                   edgecolor='darkgrey', facecolor='lightgray')
        ax.add_patch(region)
    
    plt.show()


def plot_map_comparison(mf, lat=(-90,90), lon=(-180,180), show=True, grid_size=0.5,
                        vmin=None, vmax=None):
    '''
    '''
    # If a list of MatchFrames is provided convert these into a single MatchFrame to plot
    if isinstance(mf, list):
        mf = aeroct.concatenate_data_frames(mf)
        date = '{0} to {1}'.format(mf.date[0].date(), mf.date[-1].date())
    else:
        date = mf.date.date()
    
    lon_mid = (lon[0] + lon[1]) / 2
    
    # Figure layout
    fig = plt.figure()
    ax = plt.axes([0.1, 0.15, 0.75, 0.75], projection=ccrs.PlateCarree())
    slider_ax = plt.axes([0.1, 0.05, 0.75, 0.03])
    colorbar_ax = plt.axes([0.88, 0.15, 0.03, 0.75])
    
    ax.set_xlim(lon)
    ax.set_ylim(lat)
    
    cmap = cm.get_cmap('Oranges')
    
    # Find the data within the given longitude and latitude bounds
    in_bounds = (mf.longitudes > lon[0]) & (mf.longitudes < lon[1]) & \
                (mf.latitudes > lat[0]) & (mf.latitudes < lat[1])
    lons = mf.longitudes[in_bounds]
    lats = mf.latitudes[in_bounds]
    
    data1 = mf.data[0, in_bounds]
    data2 = mf.data[1, in_bounds]
    plt.title('AOD difference (mean) : {0} - {1} for {2}'\
              .format(mf.names[1], mf.names[0], date))
    
    # If AERONET is included plot the sites on a map
    if np.any([mf.data_sets[i] == 'aeronet' for i in [0,1]]):
        # Get the list of data points at each location
        site_lons, i_site = np.unique(lons, return_index=True)
        site_lats = lats[i_site]
        in_sites = site_lons[:, np.newaxis] == lons
        # Average the AOD at each site and take std
        site_data1_avg = np.mean(data1 * in_sites, axis=1)
        site_data2_avg = np.mean(data2 * in_sites, axis=1)
        
        # Plot the initial scatterplots
        im = ax.scatter(site_lons, site_lats, c=site_data1_avg, s=50, cmap=cmap,
                         vmin=vmin, vmax=vmax)
        
        site_lons2 = site_lons[site_lons > lon_mid]
        site_lats2 = site_lats[site_lons > lon_mid]
        site_data2 = site_data2_avg[site_lons > lon_mid]
        sc = ax.scatter(site_lons2, site_lats2, c=site_data2, s=50, cmap=cmap,
                         vmin=vmin, vmax=vmax)
        
        def update(val):
            lon_var = slide_lon.val
            
            show_site2 = site_lons > lon_var
            site_lons2 = site_lons[show_site2]
            site_lats2 = site_lats[show_site2]
            site_data2 = site_data2_avg[show_site2]
            site_lons1 = site_lons[~show_site2]
            site_lats1 = site_lats[~show_site2]
            site_data1 = site_data2_avg[~show_site2]
            
            # Plot the updated scatter plots
            if site_data1.size > 0:
                ax.scatter(site_lons1, site_lats1, c=site_data1, s=50, cmap=cmap,
                         vmin=vmin, vmax=vmax)
            if site_data1.size > 1:
                ax.scatter(site_lons2, site_lats2, c=site_data2, s=50, cmap=cmap,
                         vmin=vmin, vmax=vmax)
            
            vline.set_xdata([lon_var, lon_var])
            fig.canvas.draw_idle()
    
    # OTHERWISE PLOT A GRID
    else:
        # Using scipy.interpolate.griddata
        # First get the axes
        grid = np.mgrid[(lon[0] + grid_size/2) : lon[1] : grid_size,
                        (lat[0] + grid_size/2) : lat[1] : grid_size]
        ll = zip(lons, lats)
        
        data_grid = griddata(ll, data1, tuple(grid), method='linear')
        
        # Mask grid data where there are no nearby points. Firstly create kd-tree
        THRESHOLD = grid_size   # Maximum distance to look for nearby points
        tree = cKDTree(ll)
        xi = _ndim_coords_from_arrays(tuple(grid))
        dists = tree.query(xi)[0]
        # Copy original result but mask missing values with NaNs
        data_grid[dists > THRESHOLD] = np.nan
        
        plt.pcolormesh(grid[0], grid[1], data_grid, cmap=cmap,
                       vmin=vmin, vmax=vmax)
    
    # Slider
    slide_step = (lon[1] - lon[0]) / 20
    slide_lon = widgets.Slider(slider_ax, 'Move ->', lon[0], lon[1], valinit=lon_mid,
                               valstep=slide_step, facecolor='white')
    vline, = ax.plot([lon_mid, lon_mid], ax.get_ylim(), c='k', linewidth=5)
    slide_lon.on_changed(update)
    
    ax.coastlines()
    plt.colorbar(im, cax=colorbar_ax)
    if show == True:
        plt.show()
        return
    else:
        return fig


def animate_cube(df):
    
    fig, ax = plt.subplots()
    im = iplt.pcolormesh(df.cube[0])
    plt.colorbar(im)
    
    def animate(i):
        C = df.cube.data[i]
        C = C[:-1, :-1] # Necessary for shading='flat'
        im.set_array(C.ravel())
        return im,
    
    ani = animation.FuncAnimation(fig, animate, interval=1000, blit=True, save_count=len(df.cube.data)-1)
    
    plt.show()
