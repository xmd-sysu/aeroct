'''
This module contains the functions required to match up data from two data frames in both
time and space.

Created on Jun 27, 2018

@author: savis

TODO: Add regridding of one cube to the other in model-model match-up
'''
from __future__ import division, print_function
import warnings
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from data_frame import MatchFrame

div0 = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=(b != 0))


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


def sat_anet_match(df_s, df_a, match_time, match_rad, min_points=2):
    '''
    Return the AOD average, standard deviation, number of points, longitude, latitude
    and time for each matched pair. There is a match if a satellite data point is within
    match_time and match_rad of an AERONET data point.
    
    Parameters:
    df_s : AeroCT DataFrame
        The data-frame obtained with aeroct.load() containing total AOD satellite data.
    df_a : AeroCT DataFrame
        The data-frame obtained with aeroct.load() containing AERONET data.
    match_time : float
        The time over which data will be matched and averaged in minutes.
    match_rad : int
        The radius for which data will be matched and averaged in degrees.
    min_points : int, optional (Default: 2)
        The minimum number of points from both df1 and df2 in a given matched data point
        that is required to store that data point. 
    '''
    if (df_s.aod[0] is None) | (df_a.aod[0] is None):
        raise ValueError('Both data frames must have total AOD data.')
    
    K = 10  # Number of nearest neighbours to find of each site
    
    # Bin the time
    t_mult = 60 / match_time
    s_time = np.rint(df_s.times * t_mult) / t_mult
    a_time = np.rint(df_a.times * t_mult) / t_mult
    
    # Get the list of times included in both sets of data
    times = []
    for t in np.unique(s_time):
        if np.any(a_time == t):
            times.append(t)
    times = np.array(times)
    
    # AVERAGE AERONET SITES
    # This gives an array for the average AOD at each AERONET location for each time
    # Index 1: time, index 2: location (ordered by np.unique(df_a.latitudes))
    site_lats = np.unique(df_a.latitudes)[:, np.newaxis]
    a_aod_avg = np.zeros((times.size, site_lats.size))
    a_aod_std = np.zeros((times.size, site_lats.size))
    a_aod_num = np.zeros((times.size, site_lats.size))
    # To get average difference in time for each site
    a_time_avg = np.zeros((times.size, site_lats.size))
    
    for i_t, t in enumerate(times):
        a_lats_t = df_a.latitudes[a_time == t]
        # from_site is a matrix of booleans, the 1st index: site, 2nd: data point
        from_site = (a_lats_t == site_lats.repeat(len(a_lats_t), axis=1))
        site_aod = np.ma.masked_where(~from_site, df_a.aod[0][a_time == t] * from_site)
        site_times = np.ma.masked_where(~from_site, df_a.times[a_time == t] * from_site)
        
        a_aod_avg[i_t] = np.mean(site_aod, axis=1)
        a_aod_std[i_t] = np.std(site_aod, axis=1)
        a_aod_num[i_t] = np.sum(from_site, axis=1)
        a_time_avg[i_t] = np.mean(site_times, axis=1)
        
        # Set the standard deviation of the sites with just one measurement to be equal
        # to the average of the standard deviations with more than one measurement
        a_aod_std[a_aod_num == 1] = np.mean(a_aod_std[a_aod_num > 1])
    
    lats, i_loc = np.unique(df_a.latitudes, return_index=True)
    lons = df_a.longitudes[i_loc]
    
    # INCLUDE ONLY THE SITES WITH NEARBY SATELLITE DATA
    # For each aeronet location find the nearest neighbour
    s_ll = np.array([df_s.longitudes, df_s.latitudes]).T
    a_ll = np.array([lons, lats]).T
    dist = getnn(s_ll, a_ll, match_rad, k=1)[1]
    
    # Ignore sites with no nearest neighbour
    has_neighbour = np.isfinite(dist[:])
    a_aod_avg = a_aod_avg[:, has_neighbour]
    a_aod_std = a_aod_std[:, has_neighbour]
    a_aod_num = a_aod_num[:, has_neighbour]
    a_time_avg = a_time_avg[:, has_neighbour]
    lons = lons[has_neighbour]
    lats = lats[has_neighbour]
    a_ll = a_ll[has_neighbour]
    
    # FOR EACH REMAINING SITE FIND THE AVERAGE SATELLITE AOD AT EACH TIME
    s_aod_avg = np.zeros_like(a_aod_avg)
    s_aod_std = np.zeros_like(a_aod_std)
    s_aod_num = np.zeros_like(a_aod_num)
    s_time_avg = np.zeros_like(a_time_avg)
    
    for i_t, t in enumerate(times):
        # Add nan to prevent out of array references from getnn()
        s_aod_t = np.append(df_s.aod[0][s_time == t], np.nan)
        s_real_times_t = np.append(df_s.times[s_time == t], np.nan)
        s_ll_t = s_ll[s_time == t]
        
        # For each site find the indices of the nearest 10 satellite points within
        # match_rad using cKDTree.
        s_nn_idx, dist = getnn(s_ll_t, a_ll, match_rad, k=K)
        
        # Suppress warnings from averaging over empty arrays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # AVERAGE SATELLITE DATA
            s_aod_avg[i_t] = np.nanmean(s_aod_t[s_nn_idx], axis=1)
            s_aod_std[i_t] = np.nanstd(s_aod_t[s_nn_idx], axis=1)
            s_aod_num[i_t] = np.sum(np.isfinite(dist), axis=1)
            s_time_avg[i_t]= np.nanmean(s_real_times_t[s_nn_idx], axis=1)
    
    # REARANGE THE OUTPUT TO BE IN THE REQUIRED FORMAT
    # Reshape the longitudes, latitudes, and times to be the shape of the grid
    [i_1, i_2] = np.indices(a_aod_avg.shape)
    times = times[i_1]
    lons, lats = lons[i_2], lats[i_2]
    
    # Unravel the arrays
    avg = np.array([s_aod_avg.ravel(), a_aod_avg.ravel()])
    std = np.array([s_aod_std.ravel(), a_aod_std.ravel()])
    num = np.array([s_aod_num.ravel(), a_aod_num.ravel()])
    time_diff = np.array(a_time_avg - s_time_avg).ravel()
    times = times.ravel()
    lons, lats = lons.ravel(), lats.ravel()
    
    # Return only elements for which there is enough of both satellite and AERONET data
    r = (num[0] >= min_points) & (num[1] >= min_points)
    return [avg[:, r], std[:, r], num[:, r], time_diff[r], lons[r], lats[r], times[r]]


def model_anet_match(df_m, df_a, match_time, match_rad, min_points=2):
    '''
    Return the AOD average, standard deviation, number of points, longitude, latitude
    and time for each matched pair. There is a match if a model data point is within
    match_time and match_rad of an AERONET data point.
    
    Parameters:
    df_m : AeroCT DataFrame
        The data-frame obtained with aeroct.load() containing model data.
    df_a : AeroCT DataFrame
        The data-frame obtained with aeroct.load() containing AERONET data.
    match_time : float
        The time over which data will be matched and averaged in minutes.
    match_rad : int
        The radius for which data will be matched and averaged in degrees.
        (Only accurate for less than ~2.5)
    min_points : int, optional (Default: 2)
        The minimum number of points from both df1 and df2 in a given matched data point
        that is required to store that data point. 
    '''
    
    if ((df_m.aod[1] is None) & (df_m.dust_filters is None)) | \
       ((df_a.aod[1] is None) & (df_a.dust_filters is None)):
        raise ValueError('Both data frames must have dust AOD data.')
    
    K = 10  # Number of nearest neighbours to find for each site
    
    # Bin the time
    t_mult = 60 / match_time
    m_time = np.rint(df_m.times * t_mult) / t_mult
    a_time = np.rint(df_a.times * t_mult) / t_mult
    
    # Get the list of times included in both sets of data
    times = []
    for t in np.unique(m_time):
        if np.any(a_time == t):
            times.append(t)
    times = np.array(times)
    
    # AVERAGE AERONET SITES
    # This gives an array for the average AOD at each AERONET location for each time
    # Index 1: time, index 2: location (ordered by np.unique(df_a.latitudes))
    site_lats = np.unique(df_a.latitudes)[:, np.newaxis]
    a_aod_avg = np.zeros((times.size, site_lats.size))
    a_aod_std = np.zeros((times.size, site_lats.size))
    a_aod_num = np.zeros((times.size, site_lats.size))
    # To get average difference in time for each site
    a_time_avg = np.zeros((times.size, site_lats.size))
    
    for i_t, t in enumerate(times):
        a_lats_t = df_a.latitudes[a_time == t]
        # from_site is a matrix of booleans, the 1st index: site, 2nd: data point
        from_site = (a_lats_t == site_lats.repeat(len(a_lats_t), axis=1))
        site_aod = np.ma.masked_where(~from_site, df_a.aod[1][a_time == t] * from_site)
        site_times = np.ma.masked_where(~from_site, df_a.times[a_time == t] * from_site)
        
        a_aod_avg[i_t] = np.mean(site_aod, axis=1)
        a_aod_std[i_t] = np.std(site_aod, axis=1)
        a_aod_num[i_t] = np.sum(from_site, axis=1)
        a_time_avg[i_t] = np.mean(site_times, axis=1)
        
        # Set the standard deviation of the sites with just one measurement to be equal
        # to the average of the standard deviations with more than one measurement
        a_aod_std[a_aod_num == 1] = np.mean(a_aod_std[a_aod_num > 1])
    
    lats, i_loc = np.unique(df_a.latitudes, return_index=True)
    lons = df_a.longitudes[i_loc]
    a_ll = zip(lons, lats)
    
    # FOR EACH SITE FIND THE AVERAGE MODEL AOD AT EACH TIME
    
    # Get a 1Dx2 list of latitudes and longitudes of the 50x50 grid points around each
    # AERONET site so that it may be passed to getnn()
    # Firstly we will get the indices
    N = 50
    lon_diffs = np.abs(lons[:, np.newaxis] - df_m.longitudes)
    site_lon_idx = np.argmin(lon_diffs, axis=1)
    lat_diffs = np.abs(lats[:, np.newaxis] - df_m.latitudes)
    site_lat_idx = np.argmin(lat_diffs, axis=1)
    site_idx_grid = np.array([np.mgrid[idx[0]-N/2 : idx[0]+N/2+1, idx[1]-N/2 : idx[1]+N/2+1]
                              for idx in zip(site_lon_idx, site_lat_idx)], dtype=np.int)
    m_lon_idx = site_idx_grid[:, 0]
    m_lat_idx = site_idx_grid[:, 1]
    # Remove negative indices and too large indices
    positive = (m_lon_idx >= 0) & (m_lat_idx >= 0)
    below_max = (m_lon_idx < df_m.longitudes.size) & (m_lat_idx < df_m.latitudes.size)
    m_lon_idx = m_lon_idx[positive & below_max]
    m_lat_idx = m_lat_idx[positive & below_max]
    
    # Get the latitudes, longitudes, aod data, and real times
    m_lons = df_m.longitudes[m_lon_idx]
    m_lats = df_m.latitudes[m_lat_idx]
    m_ll = zip(m_lons, m_lats)
    m_aod = df_m.aod[1][:, m_lat_idx, m_lon_idx]
    
    # Now find the nearest neighbours and average
    m_aod_avg = np.zeros_like(a_aod_avg)
    m_aod_std = np.zeros_like(a_aod_std)
    m_aod_num = np.zeros_like(a_aod_num)
    m_real_times = np.zeros_like(a_aod_avg)
    
    for i_t, t in enumerate(times):
        # Add nan to prevent out of array references from getnn()
        m_aod_t = np.append(m_aod[i_t], np.nan)
        
        # For each site find the indices of the nearest 10 satellite points within
        # match_rad using cKDTree.
        m_nn_idx, dist = getnn(m_ll, a_ll, match_rad, k=K)
        
        # Suppress warnings from averaging over empty arrays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # AVERAGE SATELLITE DATA
            m_aod_avg[i_t] = np.nanmean(m_aod_t[m_nn_idx], axis=1)
            m_aod_std[i_t] = np.nanstd(m_aod_t[m_nn_idx], axis=1)
            m_aod_num[i_t] = np.sum(np.isfinite(dist), axis=1)
            m_real_times[i_t] = np.full_like(m_aod_avg[i_t], df_m.times[m_time==t])
    
    # REARANGE THE OUTPUT TO BE IN THE REQUIRED FORMAT
    # Reshape the longitudes, latitudes, and times to be the shape of the grid
    [i_1, i_2] = np.indices(a_aod_avg.shape)
    times = times[i_1]
    lons, lats = lons[i_2], lats[i_2]
    
    # Unravel the arrays
    avg = np.array([m_aod_avg.ravel(), a_aod_avg.ravel()])
    std = np.array([m_aod_std.ravel(), a_aod_std.ravel()])
    num = np.array([m_aod_num.ravel(), a_aod_num.ravel()])
    time_diff = np.array(a_time_avg - m_real_times).ravel()
    times = times.ravel()
    lons, lats = lons.ravel(), lats.ravel()
    
    # Return only elements for which there is enough of both satellite and AERONET data
    r = (num[0] >= min_points) & (num[1] >= min_points)
    return [avg[:, r], std[:, r], num[:, r], time_diff[r], lons[r], lats[r], times[r]]


def model_sat_match(df_m, df_s, match_time, match_dist, min_points=2, limits=(-180, 180, -90, 90)):
    '''
    Return the AOD average, standard deviation, number of points, longitude, latitude
    and time for each matched pair. Each matched pair has model data within match_time
    of the satellite data and the data is averaged on a grid with grid size match_rad.
    
    Parameters:
    df_m : AeroCT DataFrame
        The data-frame obtained with aeroct.load() containing model data.
    df_s : AeroCT DataFrame
        The data-frame obtained with aeroct.load() containing coarse mode satellite data.
    match_time : float
        The time over which data will be matched and averaged in minutes.
    match_dist : int
        The size of the grid cells for which data will be matched and averaged in degrees.
    min_points : int, optional (Default: 2)
        The minimum number of points from both df1 and df2 in a given matched data point
        that is required to store that data point. 
    '''
    
    if ((df_m.aod[1] is None) & (df_m.dust_filters is None)) | \
       ((df_s.aod[1] is None) & (df_s.dust_filters is None)):
        raise ValueError('Both data frames must have dust AOD data.')
    
    # Include only dust AOD data
    s_aod, s_lons, s_lats, s_real_times = df_s.get_data(aod_type='dust')
    
    # Take only data within the given limits
    lon_restriction = (s_lons > limits[0]) & (s_lons < limits[1])
    lat_restriction = (s_lats > limits[2]) & (s_lats < limits[3])
    s_aod = s_aod[lon_restriction & lat_restriction]
    s_lons = s_lons[lon_restriction & lat_restriction]
    s_lats = s_lats[lon_restriction & lat_restriction]
    s_real_times = s_real_times[lon_restriction & lat_restriction]
    
    # Bin the time
    t_mult = 60 / match_time
    m_times = np.rint(df_m.times * t_mult) / t_mult
    s_times = np.rint(s_real_times * t_mult) / t_mult
    
    # Get the list of times included in both sets of data
    times = []
    for t in np.unique(m_times):
        if np.any(s_times == t):
            times.append(t)
    times = np.array(times)
    
    # Firstly bin the data into a grid of latitude and longitude offset from the dateline
    s_lons = np.rint((s_lons - match_dist/2) / match_dist) * match_dist + match_dist/2
    s_lats = np.rint(s_lats / match_dist) * match_dist
    s_ll = np.array([s_lons, s_lats])
    m_lons = np.rint((df_m.longitudes - match_dist/2) / match_dist) * match_dist + match_dist/2
    m_lats = np.rint(df_m.latitudes / match_dist) * match_dist
    # Ensure the longitudes go from -180 to 180 so that lon_uniq_mask works correctly
    lon_sort_idx = np.argsort(m_lons)
    m_lons = m_lons[lon_sort_idx]
    m_aod = df_m.aod[1][:, :, lon_sort_idx]
    
    lons, lats, times_arr = [], [], []
    m_aod_avg, m_aod_std, m_aod_num = [], [], []
    s_aod_avg, s_aod_std, s_aod_num = [], [], []
    time_diff = []
    
    for time in times:
        s_ll_t = s_ll[:, s_times==time]
        s_aod_t = s_aod[s_times==time]
        m_aod_t = m_aod[m_times==time].ravel()
        s_real_times_t = s_real_times[s_times==time]
         
        # FIND THE MEAN AND STANDARD DEVIATION OF THE SATELLITE DATA IN EACH GRID CELL
        # Sort the latitudes and longitudes to bring duplicate locations together
        sort_idx = np.lexsort(s_ll_t)
        sorted_ll = s_ll_t[:, sort_idx]
        sorted_aod = s_aod_t[sort_idx]
        sorted_times = s_real_times_t[sort_idx]
         
        # Obtain a mask to indicate where the new locations begin
        uniq_mask = np.append(True, np.any(np.diff(sorted_ll, axis=1), axis=0))
         
        # Find an ID number to identify the location of each member of the sorted arrays
        ID = np.cumsum(uniq_mask) - 1
         
        # Take the average and standard deviations for each location
        s_aod_avg_t = div0(np.bincount(ID, sorted_aod), np.bincount(ID))
        s_aod_std_t = np.sqrt(np.abs(div0(np.bincount(ID, sorted_aod**2),
                                          np.bincount(ID)) - s_aod_avg_t**2))
        s_aod_num_t = np.bincount(ID)
        s_time_avg_t = div0(np.bincount(ID, sorted_times), np.bincount(ID))
         
        lons_t = sorted_ll[0, uniq_mask]
        lats_t = sorted_ll[1, uniq_mask]
         
        # NOW FIND THE MEAN AND STD OF THE MODEL DATA IN EACH GRID CELL
        lon_uniq_mask = np.append(True, np.diff(m_lons) > 0)
        lat_uniq_mask = np.append(True, np.diff(m_lats) > 0)
         
        # ID of each lon, lat, and lon-lat pair
        lon_ID = np.cumsum(lon_uniq_mask) - 1
        lat_ID = np.cumsum(lat_uniq_mask) - 1
        ll_ID = (lon_ID + lat_ID[:, np.newaxis] * (np.max(lon_ID) + 1)).ravel()
         
        # Average and std of each grid cell
        m_aod_avg_t = div0(np.bincount(ll_ID, m_aod_t), np.bincount(ll_ID))
        m_aod_std_t = np.sqrt(np.abs(div0(np.bincount(ll_ID, m_aod_t**2),
                                          np.bincount(ll_ID)) - m_aod_avg_t**2))
             
        m_aod_num_t = np.bincount(ll_ID)
        
        # Longitudes and latitudes of each grid cell
        m_lon_uniq = m_lons[lon_uniq_mask]
        m_lat_uniq = m_lats[lat_uniq_mask]
        m_grid_lon = m_lon_uniq[np.newaxis, :].repeat(m_lat_uniq.size, axis=0).ravel()
        m_grid_lat = m_lat_uniq[:, np.newaxis].repeat(m_lon_uniq.size, axis=1).ravel()
        
        # TAKE THE MODEL DATA FOR EVERY CELL FILLED WITH SATELLITE DATA
        # Find model locations in the satellite grid locations list
        m_loc_comparator = m_grid_lon + m_grid_lat * 10000
        s_loc_comparator = lons_t + lats_t * 10000
        in_loc_t = np.in1d(m_loc_comparator, s_loc_comparator)
        
        # Get model data in each satellite location
        m_aod_avg_t = m_aod_avg_t[in_loc_t]
        m_aod_std_t = m_aod_std_t[in_loc_t]
        m_aod_num_t = m_aod_num_t[in_loc_t]
        
#         # Troubleshooting
#         if time == 12.0:
#             a = np.array([m_grid_lon[in_loc_t], m_grid_lat[in_loc_t]])
#             a = a[:,np.lexsort(a)]
#             print(a[:,a[0]==0])
#             print(a[:,np.append(False,np.all(np.diff(a, axis=1)==0, axis=0))])
#             print(len(m_aod_avg_t))
         
        # APPEND THE DATA
        lons.extend(lons_t)
        lats.extend(lats_t)
        times_arr.extend(np.full_like(s_aod_avg_t, time))
        s_aod_avg.extend(s_aod_avg_t)
        s_aod_std.extend(s_aod_std_t)
        s_aod_num.extend(s_aod_num_t)
        m_aod_avg.extend(m_aod_avg_t)
        m_aod_std.extend(m_aod_std_t)
        m_aod_num.extend(m_aod_num_t)
        time_diff.extend(df_m.times[m_times==time] - s_time_avg_t)
    
    lons, lats, times_arr = np.array(lons), np.array(lats), np.array(times_arr)
    aod_avg = np.array([m_aod_avg, s_aod_avg])
    aod_std = np.array([m_aod_std, s_aod_std])
    aod_num = np.array([m_aod_num, s_aod_num])
    time_diff = np.array(time_diff)
     
    # Return only elements for which there is enough of both model and satellite data
    r = (aod_num[0] >= min_points) & (aod_num[1] >= min_points)
    return [aod_avg[:, r], aod_std[:, r], aod_num[:, r], time_diff[r],
            lons[r], lats[r], times_arr[r]]


def collocate(df1, df2, match_time=30, match_rad=25, min_points=2):
    '''
    This matches up elements in time and space from two data frames with the same date
    and wavelength. The outputs are new data frames containing the averaged AOD data
    matched up over area and time. By default the matching up is performed over a 30
    minute time frame and a radius of 25 km.
    NOTE: match_rad is converted to degrees and a circle of latitudes and longitudes are
    used. Therefore not all the data within match_rad may cause a match near the poles.
    
    Parameters:
    df1, df2 : AeroCT data frame
        The two data frames to match up.
    match_time : float, optional (Default: 30 (minutes))
        The time over which data will be matched and averaged in minutes.
    match_rad : int, optional (Default: 25 (km))
        The radius for which data will be matched and averaged in kilometers.
    min_points : int, optional (Default: 2)
        The minimum number of points from both df1 and df2 in a given matched data point
        that is required to store that data point. 
    '''
    
    forecasts = (df1.forecast_time, df2.forecast_time)
    data_sets = (df1.data_set, df2.data_set)
    
    if df1.date != df2.date:
        raise ValueError('The dates of the data frames do not match.')
    if df1.wavelength != df2.wavelength:
        raise ValueError('The wavelengths of the data frames do not match.')
    
    # Convert match_dist from km into degrees
    match_rad = np.arcsin(match_rad / 6371) * 180 / np.pi
    
    # Satellite-AERONET match-up
    if (df1.cube is None) & (df2.cube is None):
        
        if df2.data_set == 'aeronet':
            # params  has the form [aod, std, num, lon, lat, time]
            params = sat_anet_match(df1, df2, match_time, match_rad, min_points)
            
        elif df1.data_set == 'aeronet':
            params = sat_anet_match(df2, df1, match_time, match_rad, min_points)
            param012 = [params[i][::-1] for i in range(3)]
            param3 = -1 * params[3]
            param012.append(param3)
            param012.extend(params[4:])
        
        [aod, std, num, time_diff, lon, lat, time] = param012
        
        return MatchFrame(aod, std, num, time_diff, lon, lat, time, df1.date, match_time,
                          match_rad, df1.wavelength, forecasts, data_sets, aod_type=0)
    
    # Model-AERONET match-up
    elif (df1.cube != None) & (df2.data_set == 'aeronet'):
        params = model_anet_match(df1, df2, match_time, match_rad, min_points)
        
        [aod, std, num, time_diff, lon, lat, time] = params
        
        return MatchFrame(aod, std, num, time_diff, lon, lat, time, df1.date, match_time, match_rad,
                          df1.wavelength, forecasts, data_sets, aod_type=1)
    
    # Same as above but the other way around
    elif (df1.data_set == 'aeronet') & (df2.cube != None):
        params = model_anet_match(df2, df1, match_time, match_rad)
        param012 = [params[i][::-1] for i in range(3)]
        param3 = -1 * params[3]
        param012.append(param3)
        param012.extend(params[4:])
        
        [aod, std, num, time_diff, lon, lat, time] = param012
        
        return MatchFrame(aod, std, num, time_diff, lon, lat, time, df1.date, match_time, match_rad,
                          df1.wavelength, forecasts, data_sets, aod_type=1)
    
    # Model-Satellite match-up
    elif (df1.cube != None) & (df2.cube is None):
        params = model_sat_match(df1, df2, match_time, match_rad, min_points)
        
        [aod, std, num, time_diff, lon, lat, time] = params
        
        return MatchFrame(aod, std, num, time_diff, lon, lat, time, df1.date, match_time,
                          2 * match_rad, df1.wavelength, forecasts, data_sets, aod_type=1)
    
    # Same as above but the other way around
    elif (df1.cube is None) & (df2.cube != None):
        params = model_sat_match(df2, df1, match_time, match_rad, min_points)
        param012 = [params[i][::-1] for i in range(3)]
        param3 = -1 * params[3]
        param012.append(param3)
        param012.extend(params[4:])
        
        [aod, std, num, time_diff, lon, lat, time] = param012
        
        return MatchFrame(aod, std, num, time_diff, lon, lat, time, df1.date, match_time,
                          2 * match_rad, df1.wavelength, forecasts, data_sets, aod_type=1)
    
    # Model-Model match-up
    elif (df1.cube != None) & (df2.cube != None):
        
        if ((df1.aod[1] is None) & (df1.dust_filter is None)) | \
           ((df2.aod[1] is None) & (df2.dust_filter is None)):
            raise ValueError('Both data frames must have dust AOD data.')
        
        # Get the dust AOD values
        aod1, aod2 = df1.aod[1], df2.aod[1]
        
        # Get times shared between the two data frames
        in_shared_times = np.array([time in df2.times for time in df1.times])
        times = df1.times[in_shared_times]
        times1_idx = np.arange(len(df1.times))[in_shared_times]
        
        aod = np.zeros((2, len(times)) + aod1[0].shape)
        cube_data = np.zeros((len(times),) + aod1[0].shape)
        for i_t, time in enumerate(times):
            aod[0, i_t] = aod1[df1.times==time][0]
            aod[1, i_t] = aod2[df2.times==time][0]
            cube_data[i_t] = aod[1, i_t] - aod[0, i_t]
            
        # Get data to put into MatchFrame
        std = np.zeros_like(aod)        # No averaging
        num = np.ones_like(aod)         # is performed
        lon, lat = np.meshgrid(df1.longitudes, df1.latitudes)
        lon_f = np.tile(lon.ravel(), len(times))            # Same dimensions as
        lat_f = np.tile(lat.ravel(), len(times))            # flattened data
        cube = df1.cube[times1_idx]
        cube.data = cube_data   # Cube with data of df2 - df1
        
        return MatchFrame(aod, std, num, time_diff, lon_f, lat_f, times, df1.date, None, None,
                          df1.wavelength, forecasts, data_sets, aod_type=1, cube=cube)
    
    else:
        raise ValueError('Unrecognised data frame types.')


if __name__ == '__main__':
    Lon1 = np.random.random(2560)
    Lat1 = np.random.random(2560)
    Lon2 = np.random.random(206)
    Lat2 = np.random.random(206)
    d1 = np.asarray(zip(Lon1, Lat1))
    d2 = np.asarray(zip(Lon2, Lat2))
    i, d = getnn(d1, d2, 0.1, k=2)
#     print(i)
#     print(d)
#     print(d1[i])
    