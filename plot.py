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
'''
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt, cm, animation, widgets
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree

sys.path.append('/home/h01/savis/workspace/summer')
import aeroct

# Suppress warnings from importing iris.plot in python 2
import warnings
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
        raise ValueError('Only AERONET data may be used in this function.')
    
    lons, i_uniq, i_inv = np.unique(df.longitudes, return_index=True , return_inverse=True)
    lon = lons[site]
    lat = df.latitudes[i_uniq[site]]
    
    aod = df.data[i_inv == site]
    times = df.times[i_inv == site]
    
    plt.plot(times, aod)#, 'ro')
    plt.title('Daily AOD from AERONET at (lon: {:.02f}, lat: {:.02f})'.format(lon, lat))
    plt.show()


def plot_aod_hist(df, aod_type=None):
    if df.data_set == 'metum':
        if (aod_type is None) | (aod_type == 'dust'):
            aod = df.aod[1]
    else:
        aod = df.get_data(aod_type)
    
    plt.hist(aod, bins=20)
    plt.show()
    

def plot_map(df, data_type=None, lat=(-90,90), lon=(-180,180), plot_type='pcolormesh',
             show=True, grid_size=0.5, vmin=None, vmax=None):
    '''
    For DataFrames this function will plot the daily average of the AOD at individual
    sites for AERONET data, otherwise on a grid.
    For MatchFrames this the difference in AOD is plotted (data_set[1] - data_set[0]).
    This will be displayed as individual sites if AERONET data is included, otherwise on
    a grid.
    
    Parameters:
    df : AeroCT DataFrame / MatchFrame, or list of DataFrames / MatchFrames
    data_type : {None, 'total' or 'dust'} (Default: None)
        This describes which AOD data to plot if the data frame is a DataFrame instance.
        If None is chosen and the data frame only includes a single type of AOD then that
        will be plotted. If it includes both then the total AOD will be plotted. If an
        AOD type is chosen which the data frame does not include then a ValueError will
        be raised.
    lat : tuple, optional (Default: (-90, 90))
        A tuple of the latitude bounds of the plot in degrees.
    lon : tuple, optional (Default: (-180, 180))
        A tuple of the longitude bounds of the plot in degrees.
    plot_type : str, optional (Default: 'scatter')
        The type of plot to produce if it does not contain AERONET data. 'scatter' to
        plot a scatter grid of all AOD data, and 'pcolormesh' or 'contourf' for griddded
        plots.
    show : bool, optional (Default: True)
        If True the figure is shown, otherwise it is returned 
    grid_size : float, optional (Default: 1)
        The size of the grid squares in degrees if not using indivdual sites
    vmin, vmax : scalar, optional (Default: None)
        vmin and vmax are used to set limits for the color map. If either is None, it is
        autoscaled to the respective min or max of the plotted data.

    '''
    
    # Convert a list of match frames to a single match frame that may be plotted
    if isinstance(df, list):
        df = aeroct.concatenate_data_frames(df)
        date = '{0} to {1}'.format(df.date[0].date(), df.date[-1].date())
    else:
        date = df.date.date()
    
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    plt.xlim(lon)
    plt.ylim(lat)
    
    data_frame_cmap = cm.get_cmap('Oranges')
    match_frame_cmap = cm.get_cmap('RdBu_r')
    
    # USE IRIS PLOT IF THERE IS A CUBE IN THE DATA FRAME
    if type(df.cube) != type(None):
        
        day_avg_cube = df.cube.collapsed('time', analysis.MEAN)
        
        if df.__class__.__name__ == 'DataFrame':
            plt.title('{0}: AOD mean for {1}'.format(df.name, date))
            cmap = data_frame_cmap
        else:
            plt.title('AOD difference (mean) : {0} - {1} for {2}'\
                      .format(df.names[1], df.names[0], date))
            cmap = shiftedColorMap(match_frame_cmap, day_avg_cube.data, vmin, vmax)
        
        if plot_type == 'pcolormesh':
            iplt.pcolormesh(day_avg_cube, cmap=data_frame_cmap, vmin=vmin, vmax=vmax)
        elif plot_type == 'contourf':
            iplt.contourf(day_avg_cube, cmap=data_frame_cmap, vmin=vmin, vmax=vmax)
    
    elif df.__class__.__name__ == 'DataFrame':
        plt.title('{0}: AOD mean for {1}'.format(df.name, date))
        
        # Select the total or dust AOD data which is in the given bounds
        aod, lons, lats = df.get_data(data_type)[:3]
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
            
            plt.scatter(site_lons, site_lats, c=aod_site_avg, cmap=data_frame_cmap,
                        s=100, vmin=vmin, vmax=vmax)
        
        # PLOT THE AOD AT EVERY DATA POINT
        elif plot_type == 'scatter':
            plt.scatter(lons, lats, c=aod, marker='o', s=(72./fig.dpi)**2,
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
                plt.pcolormesh(grid[0], grid[1], aod_grid, cmap=data_frame_cmap,
                               vmin=vmin, vmax=vmax)
            elif plot_type == 'contourf':
                plt.contourf(grid[0], grid[1], aod_grid, cmap=data_frame_cmap,
                             vmin=vmin, vmax=vmax)
    
    elif df.__class__.__name__ == 'MatchFrame':        
        # Find the data within the given longitude and latitude bounds
        in_bounds = (df.longitudes > lon[0]) & (df.longitudes < lon[1]) & \
                    (df.latitudes > lat[0]) & (df.latitudes < lat[1])
        lons = df.longitudes[in_bounds]
        lats = df.latitudes[in_bounds]
        
        if data_type == 'time diff':
            data = df.time_diff[in_bounds]
            plt.title('Time difference (mean) : {0} - {1} for {2}'\
                      .format(df.names[1], df.names[0], date))
        else:
            data = df.data[1, in_bounds] - df.data[0, in_bounds]
            plt.title('AOD difference (mean) : {0} - {1} for {2}'\
                      .format(df.names[1], df.names[0], date))
        
        # If AERONET is included plot the sites on a map
        if np.any([df.data_sets[i] == 'aeronet' for i in [0,1]]):
            # Get the list of data points at each location
            site_lons, i_site = np.unique(lons, return_index=True)
            site_lats = lats[i_site]
            in_sites = site_lons[:, np.newaxis] == lons
            # Average the AOD at each site and take std
            site_data_avg = np.mean(data * in_sites, axis=1)
            
            # Shift colour map to have a midpoint of zero
            cmap = shiftedColorMap(match_frame_cmap, site_data_avg, vmin, vmax)
            
            plt.scatter(site_lons, site_lats, c=site_data_avg, s=100, cmap=cmap,
                        vmin=vmin, vmax=vmax)
        
        # OTHERWISE PLOT A GRID
        else:
            # Using scipy.interpolate.griddata
            # First get the axes
            grid = np.mgrid[(lon[0] + grid_size/2) : lon[1] : grid_size,
                            (lat[0] + grid_size/2) : lat[1] : grid_size]
            ll = zip(lons, lats)
            
            data_grid = griddata(ll, data, tuple(grid), method='linear')
            
            # Mask grid data where there are no nearby points. Firstly create kd-tree
            THRESHOLD = grid_size   # Maximum distance to look for nearby points
            tree = cKDTree(ll)
            xi = _ndim_coords_from_arrays(tuple(grid))
            dists = tree.query(xi)[0]
            # Copy original result but mask missing values with NaNs
            data_grid[dists > THRESHOLD] = np.nan
            
            # Shift colour map to have a midpoint of zero
            cmap = shiftedColorMap(match_frame_cmap, data_grid, vmin, vmax)
            
            if plot_type == 'pcolormesh':
                plt.pcolormesh(grid[0], grid[1], data_grid, cmap=cmap,
                               vmin=vmin, vmax=vmax)
            elif plot_type == 'contourf':
                plt.contourf(grid[0], grid[1], data_grid, cmap=cmap, vmin=vmin, vmax=vmax)
        
    
    ax.coastlines()
    plt.colorbar(orientation='horizontal')
    if show == True:
        plt.show()
        return
    else:
        return fig


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


def scatterplot(df, stats=True, log_scale=True, show=True, error=True, hm_threshold=300, **kwargs):
    '''
    This is used to plot AOD data from two sources which have been matched-up on a
    scatter plot. The function returns the figure if show=True.
    
    Parameters:
    df : AeroCT MatchFrame
        The data frame containing collocated data for a day.
    stats : bool, optional (Default: True)
        Choose whether to show statistics on the plot.
    log_scale: bool, optional (Default: True)
        Choose whether to plot the data on a log scale (if so anything below 1e-4 is not
        displayed).
    show : bool, optional (Default: True)
        If True, the plot is shown otherwise the figure is passed as an output.    
    error : bool, optional (Default: True)
        If True, error bars for the standard deviations are included on the plot.
    hm_threshold : int, optional (Default: 300)
        The threshold of number of data points above which a heat map will be plotted
        instead of a scatter plot.
    **kwargs : optional
        Arguments for the style of the scatter plot. By default c='r', marker='o',
        linestyle='None' and, if error=True, ecolor='gray'.
    '''
    if isinstance(df, list):
        df = aeroct.concatenate_data_frames(df)
        date_str = '{0} to {1}'.format(df.date[0].date(), df.date[-1].date())
    else:
        date_str = '{0}'.format(df.date[0].date())
    
    if (df.__class__.__name__ != 'MatchFrame') | (len(df.data_sets) != 2):
        raise ValueError('The data frame is unrecognised. It must be collocated data \
                         from two data sets.')
        
    # Plot a heat map if there are more data points than hm_threshold
    heatmap = (df.data[0].size > hm_threshold)
    
    fig = plt.figure()
    
    # Axes locations and sizes
    x0, y0 = 0.13, 0.08
    width, height = 0.7, 0.6
    width2, height2 = 0.1, 0.1
    
    if heatmap:
        cheight, cpad = 0.03, 0.1
        cax = fig.add_axes([x0, y0, width, cheight])
        y1 = y0 + cheight + cpad
    else:
        y1 = y0
    
    ax = fig.add_axes([x0, y1, width, height])
    ax_x = fig.add_axes([x0, y1 + height + 0.01, width, height2], sharex=ax)
    ax_y = fig.add_axes([x0 + width + 0.01, y1, width2, height], sharey=ax)
    
    # Grid cells for heat-map / histograms
    x_min, x_max = np.min(df.data[0]), np.max(df.data[0])
    y_min, y_max = np.min(df.data[1]), np.max(df.data[1])
    if log_scale:
        x_grid = 10 ** np.linspace(-4, np.log10(x_max), 101)
        y_grid = 10 ** np.linspace(-4, np.log10(y_max), 101)
    else:
        x_grid = np.linspace(x_min, x_max, 101)
        y_grid = np.linspace(y_min, y_max, 101)
    
    # Plot a scatter plot if there are fewer data points than hm_threshold
    if not heatmap:
        kwargs.setdefault('c', 'r')
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        
        if error == True:
            kwargs.setdefault('ecolor', 'gray')
            ax.errorbar(df.data[0], df.data[1], df.data_std[1], df.data_std[0], **kwargs)
        else:
            ax.plot(df.data[0], df.data[1], **kwargs)
    
    # Otherwise plot a heat-map
    else:
        # Find the number of points in each grid cell and mask those with none
        heatmap_grid = np.histogram2d(df.data[0], df.data[1], [x_grid, y_grid])[0]
        heatmap_grid = np.ma.masked_where(heatmap_grid==0, heatmap_grid)
        
        im = ax.pcolormesh(x_grid, y_grid, heatmap_grid.T, cmap='CMRmap')
        plt.colorbar(im, cax=cax, orientation='horizontal')
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(1e-4, 10, 101)
    y = df.R_INTERCEPT + x * df.R_SLOPE
    ax.plot(x, y, 'g-.', lw=2, label='Regression') # Regression line
    ax.plot(x, x, c='gray', ls='--', lw=2, label='y = x') # y = x line
    
    # Title, axes, and legend
    if df.aod_type == 0:
        aod_str = 'total AOD'
    elif df.aod_type == 1:
        aod_str = 'dust AOD'
    title = 'Collocated {0} comparison for {1}'.format(aod_str, date_str)
    fig.text(0.5, (y1 + height + height2 + 0.03), title, ha='center', fontsize=14)
    ax.legend(loc=4)
    ax.set_xlabel('{0} AOD'.format(df.names[0]))
    ax.set_ylabel('{0} AOD'.format(df.names[1]))
    
    if log_scale:
        ax.loglog()
        ax.set_xlim((1e-4, xlim[1]))
        ax.set_ylim((1e-4, ylim[1]))
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    # Ticks
    ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
    ax_x.tick_params(direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
    ax_y.tick_params(direction='in', bottom=True, top=True, left=True, right=True, labelleft=False)
    
    # Histograms
    ax_x.hist(df.data[0], bins=x_grid, color='k')
    ax_y.hist(df.data[1], bins=y_grid, color='k', orientation='horizontal')
    
    # Stats
    if stats == True:
        box = dict(facecolor='w', edgecolor='w', pad=-0.75)
        rms_str = 'RMS: {:.02f}'.format(df.RMS)
        plt.text(0.03, 0.94, rms_str, fontsize=12, transform=ax.transAxes, bbox=box)
        bias_mean_str = 'Bias mean: {:.02f}'.format(df.BIAS_MEAN)
        plt.text(0.03, 0.88, bias_mean_str, fontsize=12, transform=ax.transAxes, bbox=box)
        bias_std_str = 'Bias std: {:.02f}'.format(df.BIAS_STD)
        plt.text(0.03, 0.82, bias_std_str, fontsize=12, transform=ax.transAxes, bbox=box)
        r_str = 'Pearson R: {:.02f}'.format(df.R)
        plt.text(0.35, 0.94, r_str, fontsize=12, transform=ax.transAxes, bbox=box)
        slope_str = 'Slope: {:.02f}'.format(df.R_SLOPE)
        plt.text(0.35, 0.88, slope_str, fontsize=12, transform=ax.transAxes, bbox=box)
        intercept_str = 'Intercept: {:.02f}'.format(df.R_INTERCEPT)
        plt.text(0.35, 0.82, intercept_str, fontsize=12, transform=ax.transAxes, bbox=box)
    
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


def period_bias_plot(mf_list, show=True, **kw):
    '''
    Given a list containing MatchFrames the bias between the two sets of collocated AOD
    values are calculated. The mean bias for each day is plotted with an error bar
    containing the standard deviation of the bias.
    
    Parameters:
    mf_list : iterable of MatchFrames
        May be obtained using the period_download_and_match() function. The bias is
        the second data set AOD subtract the first.
    show : bool, optional (Default: True)
        Choose whether to show the plot. If False the figure is returned by the function.
    kwargs : optional
        These kwargs are passed to matplotlib.pyplot.errorbar() to format the plot. If
        none are supplied then the following are used:
        fmt='r.', ecolor='gray', capsize=0.
    '''
    bias_arrays = np.array([mf.data_f[1] - mf.data_f[0] for mf in mf_list])
    bias_mean = np.mean(bias_arrays, axis=1)
    bias_std = np.std(bias_arrays, axis=1)
    date_list = [mf.date for mf in mf_list]
    
    # Plot formatting
    kw.setdefault('fmt', 'r.')
    kw.setdefault('ecolor', 'gray')
    kw.setdefault('capsize', 0)
    
    fig = plt.figure()
    plt.errorbar(date_list, bias_mean, bias_std, **kw)
    
    if show == True:
        plt.show()
    else:
        return fig