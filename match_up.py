'''
This module contains the functions required to match up data from two data frames in both
time and space. 

Created on Jun 27, 2018

@author: savis

TODO: Allow match-up with gridded data.
'''
from __future__ import division, print_function
import numpy as np
from data_frame import MatchFrame


is_equal = lambda a, b: ((a - b) < a / 10000) & ((b - a) < a / 10000)


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
    if df1.latitudes.size > df2.latitudes.size:
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
    df1, df2: (AeroCT data frame) The two data frames to match up
    time_length: (float, optional) The timeframe over which data will be matched and
        averaged in hours.    Default: 0.5 (hours)
    match_dist: (int, optional) The radius for which data will be matched and averaged in
        kilometers.    Default: 25 (km)
    min_measurements: (int, optional) The minimum number of measurements required to take
        into account the average.    Default: 5
    '''
    
    if df1.date != df2.date:
        raise ValueError, 'The dates of the data frames do not match.'
    if df1.wavelength != df2.wavelength:
        raise ValueError, 'The wavelengths of the data frames do not match.'
    
    times, i_time1_bins, i_time2_bins = match_time(df1, df2, time_length)
    
    # The aod lists will be turned into a 2D numpy array. The first index will give the
    # data set and the second will give the match-up pair. The time_list will give the
    # times of each pair and the location lists will give the corresponding locations.
    aod_list, std_list, num_list = [], [], []
    lat_list, lon_list, time_list  = [], [], []
    
    for i_t, time in enumerate(times):
        # Indices for the data in each time bin
        i_t1, i_t2 = i_time1_bins[i_t], i_time2_bins[i_t]
#         print('Matching data: {:.1f}% complete.'.format(time / times[-1] * 100), end="\r")
        print('.', end='')
        
        if (df1.grid is False) & (df2.grid is False):
            # Get match-up pairs and their indices
            lat1, lon1 = df1.latitudes, df1.longitudes
            lat2, lon2 = df2.latitudes, df2.longitudes
            i1, i2 = match_loc_ungridded(lat1[i_t1], lon1[i_t1], lat2[i_t2], lon2[i_t2],
                                        match_dist, i_t1, i_t2)
            
            aod, std, num, lat, lon, time = average_aod(df1, df2, i1, i2, time,
                                                        min_measurements)
            
            aod_list.extend(aod)
            std_list.extend(std)
            num_list.extend(num)
            lat_list.extend(lat)
            lon_list.extend(lon)
            time_list.extend(time)            
        
        elif (df1.gridded is False) & (df2.gridded is True):
            pass
        elif (df1.gridded is True) & (df2.gridded is False):
            # Same as above but the other way around
            pass
        elif (df1.gridded is True) & (df2.gridded is True):
            pass
    
    print()
    aod = np.array(aod_list).T
    std = np.array(std_list).T
    num = np.array(num_list).T
    lat = np.array(lat_list)
    lon = np.array(lon_list)
    times = np.array(time_list)
    forecasts = (df1.forecast_time, df2.forecast_time)
    data_sets = (df1.data_set, df2.data_set)
    return MatchFrame(aod, std, num, lat, lon, times, df1.date, df1.wavelength, forecasts, data_sets)
    


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

if __name__ == '__main__':
    pass