'''
This module contains the functions required to match up data from two data frames in both
time and space. 

Created on Jun 27, 2018

@author: savis

TODO: Allow match-up with gridded data.
'''
from __future__ import division, print_function
import numpy as np
import numpy.ma as ma
from scipy.spatial import cKDTree
import sys
from data_frame import MatchFrame


def getnn(d1, d2, r, k=5):
    """
    Search nearest neighbors between two coordinate catalogues. See
    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.spatial.cKDTree.query.html

    Parameters
    ----------
    d1, d2 : array_like, last dimension self.m
        Arrays of points to query (in (n, 2)).

    r : nonnegative float
        Return only neighbors within this distance. This is used to prune
        tree searches, so if you are doing a series of nearest-neighbor
        queries, it may help to supply the distance to the nearest neighbor of
        the most recent point.

    k : list of integer or integer
        The list of k-th nearest neighbors to return. If k is an integer
        it is treated as a list of [1, ... k] (range(1, k+1)). Note that
        the counting starts from 1.

    Returns
    -------
    id1 : ndarray of ints
        The locations of the neighbors in self.data. If d1 has shape
        tuple+(self.m,), then id1 has shape tuple+(k,). When k == 1, the last
        dimension of the output is squeezed. Missing neighbors are indicated
        with self.n.
    d : array of floats
        The distances to the nearest neighbors. If d1 has shape tuple+(self.m,),
        then d has shape tuple+(k,). When k == 1, the last dimension of the
        output is squeezed. Missing neighbors are indicated with infinite
        distances.

    Example:
        >>> Lon1 = numpy.random.random(2000)
        >>> Lat1 = numpy.random.random(2000)
        >>> Lon2 = numpy.random.random(20)
        >>> Lat2 = numpy.random.random(20)
        >>> d1 = numpy.asarray(zip(Lon1, Lat1))
        >>> d2 = numpy.asarray(zip(Lon2, Lat2))
        >>> i, d = getnn(d1, d2, 0.1, k=3)
    """
    t = cKDTree(d1)
    d, idx = t.query(d2, k=k, eps=0, p=2, distance_upper_bound=r)
    return idx, d


def match_time(df1, df2, time_length):
    '''
    Puts the two data frames times into bins of length time_length and outputs the bins
    which are populated by both data frames. The output is a list of the times of the
    populated bins and two lists containing the indices of each dataframe's times in
    each bin.
    '''
    # Sort the data into bins of time (hours). The size of each bin is time_length.
    bins = np.arange(0, 24 + time_length/2, time_length)
    bin_bounds = np.arange(-time_length/2, 24 + time_length, time_length)[:,np.newaxis]
    time1_bin_matrix = (df1.times > bin_bounds[:-1]) & (df1.times < bin_bounds[1:])
    time2_bin_matrix = (df2.times > bin_bounds[:-1]) & (df2.times < bin_bounds[1:])
    
    #  Get a list of indices in each bin.
    i_time1 = np.arange(df1.times.size)
    i_time2 = np.arange(df2.times.size)
    i_time1_all_bins = [i_time1[time_bool] for time_bool in time1_bin_matrix]
    i_time2_all_bins = [i_time2[time_bool] for time_bool in time2_bin_matrix]
    
    # Only include bins with entries from both data frames.
    times = []
    i_time1_bins = []
    i_time2_bins = []
    for i in np.arange(bins.size):
        if (len(i_time1_all_bins[i]) != 0) & (len(i_time2_all_bins[i]) != 0):
            times.append(bins[i])
            i_time1_bins.append(i_time1_all_bins[i])
            i_time2_bins.append(i_time2_all_bins[i])
    
    return np.array(times), i_time1_bins, i_time2_bins


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def match_loc_ungridded(lat1, lon1, lat2, lon2, match_dist, i_1, i_2):
    '''
    Return the indices of matching pairs of the two data frames. lat1 and loc1 must have
    the same lengths, as must lat2 and lon2.
    '''
       
    # Obtain the pairs of indices for the two data frames which are close.
    # This will be given in loc_match_ind as: [[matching indices of 1],[" " " 2]]
    lat1 = lat1[:, np.newaxis]
    lon1 = lon1[:, np.newaxis]
    loc_match_matrix = haversine(lon1, lat1, lon2, lat2) < match_dist
    
    i_loc_matrix = np.array(np.meshgrid(i_2, i_1)) # Matrices of indices
    i_loc_match = i_loc_matrix[:,loc_match_matrix]
    
    return i_loc_match[1], i_loc_match[0]


def average_aod(df1, df2, i1, i2, time, min_meas):
    
    # The data frame with the fewest locations will be used for the locations of the
    # averaged AOD data. So need to ensure they are the right way around.
    if df1.data.size > df2.data.size:
        out = average_aod(df2, df1, i2, i1, time, min_meas)
        aod = out[0][::-1]
        std = out[1][::-1]
        num = out[2][::-1]
        return aod, std, num, out[3], out[4], out[5]
    
    lat1, lon1 = df1.latitudes, df1.longitudes
    
    lat1_uniq, i_uniq1 = np.unique(lat1[i1], return_index=True)
    lon1_uniq = lon1[i1][i_uniq1]
    
    # Take the averages and standard deviations at each unique location
    bool_matrix = (lat1[i1] == lat1_uniq[:,np.newaxis]) & (lon1[i1] == lon1_uniq[:,np.newaxis])
    aod, std, num, lat, lon = [], [], [], [], []
    for i_match in bool_matrix:
        df1_data = df1.data[i1[i_match]]
        df2_data = df2.data[i2[i_match]]
        
        # Only record values with more than min_meas measurements
        if (len(df1_data) >= min_meas) & (len(df2_data) >= min_meas):
            aod.append([np.average(df1_data), np.average(df2_data)])
            std.append([np.std(df1_data), np.std(df2_data)])
            num.append([df1_data.size, df2_data.size])
            lat.append(lat1[i1[i_match]][0])
            lon.append(lon1[i1[i_match]][0])
    time = np.full_like(lat, time)
    
    return aod, std, num, lat, lon, time


def collocate(df1, df2, time_length=0.5, match_dist=25, min_measurements=5):
    '''
    This matches up elements in time and space from two data frames with the same date
    and wavelength. The outputs are new data frames containing the averaged AOD data
    matched up over area and time. By default the matching up is performed over a 30
    minute time frame and a radius of 25 km.
    
    Parameters:
    df1, df2 : AeroCT data frame
        The two data frames to match up.
    time_length: float, optional (Default: 0.5 (hours))
        The time over which data will be matched and averaged in hours.
    match_dist : int, optional (Default: 25 (km))
        The radius for which data will be matched and averaged in kilometers.
    min_measurements : int, optional (Default: 5)
        The minimum number of measurements required to take into account the average.    
    '''
    
    forecasts = (df1.forecast_time, df2.forecast_time)
    data_sets = (df1.data_set, df2.data_set)
    
    if df1.date != df2.date:
        raise ValueError, 'The dates of the data frames do not match.'
    if df1.wavelength != df2.wavelength:
        raise ValueError, 'The wavelengths of the data frames do not match.'
    
    # Convert match_dist from km into degrees
    match_dist = np.arcsin(match_dist / 6371)
    
    times, i_time1_bins, i_time2_bins = match_time(df1, df2, time_length)
    
    if (df1.cube == None) & (df2.cube == None):
        # The aod lists will be turned into a 2D numpy array. The first index will give the
        # data set and the second will give the match-up pair. The time_list will give the
        # times of each pair and the location lists will give the corresponding locations.
        aod_list, std_list, num_list = [], [], []
        lat_list, lon_list, time_list  = [], [], []
        
        print('Matching data: ', end='')
        for i_t, time in enumerate(times):
            print('{:.1f}% '.format(time / times[-1] * 100), end='')
            
            # Data in each time bin, nan is appended for any outside index references
            # given by getnn() 
            i_t1, i_t2 = i_time1_bins[i_t], i_time2_bins[i_t]
            aod1 = np.append(df1.data[i_t1], np.nan)
            lon1 = df1.longitudes[i_t1]
            lat1 = df1.latitudes[i_t1]
            print(i_t2)
            
            # For each location of the data frame with fewer points (d2) find the indices
            # of the nearest 10 points within match_dist using cKDTree.
            df1_ll = np.array([lon1, lat1]).T
            df2_ll = np.array([df2.longitudes[i_t2], df2.latitudes[i_t2]]).T
            id1, dist = getnn(df1_ll, df2_ll, r=match_dist, k=10)
            
            # Remove any rows with no nearest neighbours
            include_row = np.isfinite(dist[:,0])
            id1 = id1[include_row]
            dist = dist[include_row]
            i_t2 = i_t2[include_row]
            aod2 = df2.data[i_t2]
            lon2 = df2.longitudes[i_t2]
            lat2 = df2.latitudes[i_t2]
            
            # Get the AOD data for each of the indices returned above and mask any values
            # for which there is not a nearest neighbour
            aod1 = ma.masked_where(~np.isfinite(dist), aod1[id1])
            
            # Take the averages and standard deviations of df2 for each location
            aod1_avg = np.nanmean(aod1, axis=1)
            aod1_std = np.nanstd(aod1, axis=1)
            aod1_num = np.sum(np.isfinite(dist), axis=1)
            
            aod_list.extend([aod1_avg, aod2])
            std_list.extend([aod1_std, np.zeros_like(aod2)])
            num_list.append([aod1_num, np.ones_like(aod2)])
            lon_list.extend(lon2)
            lat_list.extend(lat2)
            time_list.extend(np.full_like(aod2, time))
        
        print()
        aod = np.array(aod_list).T
        std = np.array(std_list).T
        num = np.array(num_list).T
        lat = np.array(lat_list)
        print(lat)
        lon = np.array(lon_list)
        times = np.array(time_list)
        cube = None
        
    elif (df1.cube != None) & (df2.cube == None):
        aod2 = np.zeros_like(df1.data)
        std2 = np.zeros_like(df1.data)
        num2 = np.zeros_like(df1.data)
        
        for i_t, time in enumerate(times):
#             print('Matching data: {:.1f}% complete.'.format(time / times[-1] * 100), end="\r")
            print('.', end='')
            
            # Indices for the data in time bin 2
            i_t2 = i_time2_bins[i_t]
            # AOD averaging over each grid cell
            #aod2[i_t], std2[i_t], num2[i_t] = one_cube_aod(df1, df2, i_t2)
        
        print()
        
        aod = np.array([df1.data, aod2])
        std = np.array([np.zeros_like(std2), std2])
        num = np.array([np.ones_like(num2), num2])
        lat = df1.latitudes
        lon = df1.longitudes
        cube = True
    
    elif (df1.cube == None) & (df2.cube != None):
        # Same as above but the other way around
        aod1 = np.zeros_like(df2.data)
        std1 = np.zeros_like(df2.data)
        num1 = np.zeros_like(df2.data)
        
        for i_t, time in enumerate(times):
#             print('Matching data: {:.1f}% complete.'.format(time / times[-1] * 100), end="\r")
            print('.', end='')
            
            # Indices for the data in each time bin 1
            i_t1 = i_time1_bins[i_t]
            # AOD averaging over each grid cell
            #aod1[i_t], std1[i_t], num1[i_t] = one_cube_aod(df2, df1, i_t1)
        
        print()
        
        aod = np.array([aod1, df2.data])
        std = np.array([std1, np.zeros_like(std1)])
        num = np.array([num1, np.ones_like(num1)])
        lat = df2.latitudes
        lon = df2.longitudes
        cube = True #!
    
    elif (df1.cube != None) & (df2.cube != None):
        pass
    
    return MatchFrame(aod, std, num, lat, lon, times, df1.date, df1.wavelength,
                      forecasts, data_sets, cube)
    


# def old_time_match():
#     # Find which data frame has the fewest time values. This data frame's times will be
#     # used for the times of the averaged AOD data.
#     if np.unique(df1.times).size < np.unique(df2.times).size:
#         time_df = 1
#     else:
#         time_df = 2
#     
#     # Obtain the pairs of indices for the two data frames which have matching times.
#     # This will be given in time_match_ind as: [[matching indices of 1],[" " " 2]]
#     times1 = df1.times[:, np.newaxis]
#     times2 = df2.times
#     time_match_matrix = (times1 - times2 < match_time) & (times2 - times1 < match_time)
#     
#     times1_ind = np.arange(times1.size)
#     times2_ind = np.arange(times2.size)
#     time_ind_matrix = np.array(np.meshgrid(times2_ind, times1_ind)) # Matrices of indices
#     time_match_ind = time_ind_matrix[:,time_match_matrix]


# def old_one_cube_aod(df1, df2, i_t2):
#     '''
#     Match up the locations of data frames 1 and 2. Data frame 1 is a cube while data
#     frame 2 is not. The third argument refers to the indices of frame 2 which match at
#     a given time. The AOD for df2 is then averaged over each grid cell.
#     Output is the average, standard deviation, and number of df2 data points in each grid
#     cell.
#     '''
#     
#     # Ensure the correct dataframes are cubes or not
#     if not (df1.cube != None) & (df2.cube == None):
#         raise TypeError, 'The data frames are of the wrong type. The first must be have \
#                           a cube while the second must not.'
#     
#     # The AOD data will be put on a grid. Firstly get the latitudes and
#     # longitudes of the grid points
#     lat_grid = df1.latitudes
#     lon_grid = df1.longitudes
#     
#     # Find if each point of data in df2 lies within each grid point. This is stored in
#     # boolean arrays for each latitude. 1st index: lon, 2nd: df2 index
#     # The bounds of each grid cell
#     lat_grid_bounds = np.zeros(lat_grid.size + 1)
#     lat_grid_bounds[1:-1] = (lat_grid[:-1] + lat_grid[1:]) / 2
#     lat_grid_bounds[[0,-1]] = 2 * lat_grid[[0,-1]] - lat_grid_bounds[[1,-2]]
#     lon_grid_bounds = np.zeros(lon_grid.size + 1)
#     lon_grid_bounds[1:-1] = (lon_grid[:-1] + lon_grid[1:]) / 2
#     lon_grid_bounds[[0,-1]] = 2 * lon_grid[[0,-1]] - lon_grid_bounds[[1,-2]]
#     
#     in_lon_grid = (df2.longitudes[i_t2] < lon_grid_bounds[1:, np.newaxis]) & \
#                   (df2.longitudes[i_t2] > lon_grid_bounds[:-1, np.newaxis])
#     
#     # Loop over each latitude
#     aod_2_avg = np.zeros((lat_grid.size, lon_grid.size))
#     aod_2_std = np.zeros((lat_grid.size, lon_grid.size))
#     aod_2_num = np.zeros((lat_grid.size, lon_grid.size))
#     for i_lat in np.arange(lat_grid_bounds.size - 1):
#         in_lat_ar = (df2.latitudes[i_t2] < lat_grid_bounds[i_lat + 1]) & \
#                     (df2.latitudes[i_t2] > lat_grid_bounds[i_lat])
#         in_grid = in_lat_ar * in_lon_grid
#         grid_data = df2.data[i_t2] * in_grid
#         grid_data = np.where(grid_data!=0, grid_data, np.nan)
#         
#         # Take the average and standard deviation of df2 AOD in each grid point
#         aod_2_avg[i_lat] = np.nanmean(grid_data, axis=1)
#         aod_2_std[i_lat] = np.nanmean(grid_data**2, axis=1) - aod_2_avg[i_lat]**2
#         aod_2_num[i_lat] = np.sum(~np.isnan(grid_data))
#     
#     return aod_2_avg, aod_2_std, aod_2_num

if __name__ == '__main__':
    Lon1 = np.random.random(2000)
    Lat1 = np.random.random(2000)
    Lon2 = np.random.random(20)
    Lat2 = np.random.random(20)
    d1 = np.asarray(zip(Lon1, Lat1))
    d2 = np.asarray(zip(Lon2, Lat2))
    i, d = getnn(d1, d2, 0.1, k=3)
    print(i)
    print(d)
    print(d1[i])