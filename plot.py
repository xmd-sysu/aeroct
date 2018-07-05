'''
Created on Jul 5, 2018

@author: savis
'''
import numpy as np
from matplotlib import pyplot as plt, colors
import iris
from iris import plot as iplt, analysis
import cartopy.crs as ccrs


def plot_anet_site(df, site=0):
    '''
    Plot the daily data for a single AERONET site.
    
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
    

def plot_map(df, lat=(-90,90), lon=(-180,180), plot_type='scatter', show=True, grid_size=1):
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
        data, 'scatter' to plot a scatter grid of all AOD data, 'grid' for a meshgrid
        plot, 'contourf' for a filled contour plot.
    show : bool, optional (Default: True)
        If True the figure is shown, otherwise it is returned 
    grid_size : float, optional (Default: 1)
        The size of the grid squares in degrees if not using indivdual sites
    '''
    
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.title('Daily AOD for {} from {}'.format(df.date.date(), df.data_set))
    plt.xlim(lon)
    plt.ylim(lat)
    cmap='Greys'
    
    if df.cube == None:
        
        if plot_type == 'sites':
            # PLOT AOD AT AERONET SITES AS A SCATTER PLOT
            # Get the list of data points at each location
            site_lons, i_site = np.unique(df.longitudes, return_index=True)
            site_lats = df.latitudes[i_site]
            in_sites = site_lons[:, np.newaxis] == df.longitudes
            # Average the AOD at each site and take std
            aod_site_avg = np.mean(df.data * in_sites, axis=1)
            aod_site_std = np.mean(df.data**2 * in_sites, axis=1) - aod_site_avg**2
            
            plt.scatter(site_lons, site_lats, c=aod_site_avg, norm=colors.LogNorm(),
                        cmap=cmap, s=100)
            ax.coastlines()
            plt.colorbar(orientation='horizontal')
            
            if show == True:
                plt.show()
                return
            else:
                return fig
        
        elif plot_type == 'scatter':
            # PLOT THE AOD AT EVERY DATA POINT
            plt.scatter(df.longitudes, df.latitudes, c=df.data,
                        marker='o', s=(72./fig.dpi)**2)
            ax.coastlines()
            plt.colorbar(orientation='horizontal')
            
            if show == True:
                plt.show()
                return
            else:
                return fig
        
        # The AOD data will be put on a grid. Firstly get the latitudes and
        # longitudes of the grid points
        lat_grid = np.arange(lat[0],  lat[1], grid_size)
        lon_grid = np.arange(lon[0], lon[1], grid_size)
        lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)
        
        # Bin the longitude and latitude
        lons = np.rint(df.longitudes / grid_size) * grid_size
        lats = np.rint(df.latitudes / grid_size) * grid_size
        
        # Find if each point of data lies within each grid point. This is stored in a
        # boolean array with indices: latitude, longitude, data frame index.            
        aod_grid_avg = np.zeros_like(lat_grid)
        aod_grid_std = np.zeros_like(lat_grid)
        for i_lat in np.arange(lon_grid[0].size):
            in_lon_mat = (lons == lon_grid[:, i_lat, np.newaxis])
            in_lat_ar = (lats == lat_grid[0, i_lat])                
            in_grid = in_lat_ar * in_lon_mat
            print(in_lat_ar)
            
            grid_data = df.data * in_grid
            grid_data = np.where(grid_data!=0, grid_data, np.nan)
            
            # Take the average and standard deviation for each grid point
            aod_grid_avg[:, i_lat] = np.nanmean(grid_data, axis=1)
#             aod_grid_std[:, i_lat] = np.nanmean(grid_data**2, axis=1) \
#                                     - aod_grid_avg[i_lat]**2
        
        if plot_type == 'grid':
            plt.pcolormesh(lon_grid, lat_grid, aod_grid_avg, norm=colors.LogNorm(),
                        cmap=cmap)
        elif plot_type == 'contourf':
            plt.contourf(lon_grid, lat_grid, aod_grid_avg, norm=colors.LogNorm(),
                        cmap=cmap)
    
    else:
        daily_average_cube = df.cube.collapsed('time', analysis.MEAN)
        if plot_type == 'grid':
            iplt.pcolormesh(daily_average_cube, norm=colors.LogNorm(), cmap=cmap)
        elif plot_type == 'contourf':
            iplt.contourf(daily_average_cube, norm=colors.LogNorm(), cmap=cmap)
    
    ax.coastlines()
    plt.colorbar(orientation='horizontal')
    plt.show()