'''
This module contains the functions required to match up data from two data frames in both
time and space. 

Created on Jun 27, 2018

@author: savis
'''

import numpy as np

def match_time(df1, df2, time_length):
    '''
    Puts the two data frames times into bins of length time_length and outputs the bins
    which are populated by both data frames. The output is a list of the times of the
    popluated bins and two lists containing the indices of each dataframe's times in
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
        if (len(i_time1_all_bins[i])==0) & (len(i_time2_all_bins[i])==0):
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


def is_close_ungridded(lat1, lon1, lat2, lon2, match_dist):
    '''
    Return the indices of 1 and the indices of 2 and the locations of whichever has fewer
    locations. lat1 and loc1 must have the same lengths, as must lat2 and lon2.
    '''
       
    # Obtain the pairs of indices for the two data frames which are close.
    # This will be given in loc_match_ind as: [[matching indices of 1],[" " " 2]]
    lat1 = lat1[:, np.newaxis]
    lon1 = lon1[:, np.newaxis]
    loc_match_matrix = haversine(lon1, lat1, lon2, lat2) < match_dist
     
    loc1_ind = np.arange(lat1.size)
    loc2_ind = np.arange(lat2.size)
    loc_ind_matrix = np.array(np.meshgrid(loc2_ind, loc1_ind)) # Matrices of indices
    loc_match_ind = loc_ind_matrix[:,loc_match_matrix]
    
    # Find which data frame has the fewest locations. This data frame's locations will be
    # used for the locations of the averaged AOD data.
    if lat1.size < lat2.size:
        i_loc1 = 
    else:
        loc_df = 2


def match_up(df1, df2, time_length=0.5, match_dist=25):
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
    '''
    
    if df1.date != df2.date:
        raise ValueError, 'The dates of the data frames do not match.'
    if df1.wavelength != df2.wavelength:
        raise ValueError, 'The wavelengths of the data frames do not match.'
    
    times, i_time1_bins, i_time2_bins = match_time(df1, df2, time_length)
    
    for i_times in np.arange(times.size):
        i_t1, i_t2 = i_time1_bins[i_times], i_time2_bins[i_times]
        if (df1.gridded is False) & (df2.gridded is False):
            lat1, lon1 = df1.latitudes[i_t1], df1.longitudes[i_t1]
            lat2, lon2 = df2.latitudes[i_t2], df2.longitudes[i_t2]
            is_close_ungridded(lat1, lon1, lat2, lon2, match_dist)
        
        elif (df1.gridded is False) & (df2.gridded is True):
            pass
        elif (df1.gridded is True) & (df2.gridded is False):
            # Same as above but the other way around
            pass
        elif (df1.gridded is True) & (df2.gridded is True):
            pass
    


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