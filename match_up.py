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
from scipy.spatial import cKDTree
from data_frame import MatchFrame
import time


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


def sat_anet_match(df_s, df_a, match_time, match_rad):
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
    '''
    if (type(df_s.aod) == type(None)) | (type(df_a.aod) == type(None)):
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
    
    for i_t, t in enumerate(times):
        a_lats_t = df_a.latitudes[a_time == t]
        # from_site is a matrix of booleans, the 1st index: site, 2nd: data point
        from_site = (a_lats_t == site_lats.repeat(len(a_lats_t), axis=1))
        site_aod = np.ma.masked_where(~from_site, df_a.aod[a_time == t] * from_site)
        
        a_aod_avg[i_t] = np.mean(site_aod, axis=1)
        a_aod_std[i_t] = np.std(site_aod, axis=1)
        a_aod_num[i_t] = np.sum(from_site, axis=1)
        
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
    lons = lons[has_neighbour]
    lats = lats[has_neighbour]
    a_ll = a_ll[has_neighbour]
    
    # FOR EACH REMAINING SITE FIND THE AVERAGE SATELLITE AOD AT EACH TIME
    s_aod_avg = np.zeros_like(a_aod_avg)
    s_aod_std = np.zeros_like(a_aod_std)
    s_aod_num = np.zeros_like(a_aod_num)
    
    for i_t, t in enumerate(times):
        # Add nan to prevent out of array references from getnn()
        s_aod_t = np.append(df_s.aod[s_time == t], np.nan)
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
    
    # REARANGE THE OUTPUT TO BE IN THE REQUIRED FORMAT
    # Reshape the longitudes, latitudes, and times to be the shape of the grid
    [i_1, i_2] = np.indices(a_aod_avg.shape)
    times = times[i_1]
    lons, lats = lons[i_2], lats[i_2]
    
    # Unravel the arrays
    avg = np.array([s_aod_avg.ravel(), a_aod_avg.ravel()])
    std = np.array([s_aod_std.ravel(), a_aod_std.ravel()])
    num = np.array([s_aod_num.ravel(), a_aod_num.ravel()])
    times = times.ravel()
    lons, lats = lons.ravel(), lats.ravel()
    
    # Return only elements for which there is both satellite and AERONET data
    r = (num[0] > 0) & (num[1] > 0)
    return [avg[:,r], std[:,r], num[:,r], lons[r], lats[r], times[r]]


def model_anet_match(df_m, df_a, match_time, match_rad):
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
    '''
    
    if (type(df_m.aod_d) == type(None)) | (type(df_a.aod_d) == type(None)):
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
    
    for i_t, t in enumerate(times):
        a_lats_t = df_a.latitudes[a_time == t]
        # from_site is a matrix of booleans, the 1st index: site, 2nd: data point
        from_site = (a_lats_t == site_lats.repeat(len(a_lats_t), axis=1))
        site_aod = np.ma.masked_where(~from_site, df_a.aod_d[a_time == t] * from_site)
        
        a_aod_avg[i_t] = np.mean(site_aod, axis=1)
        a_aod_std[i_t] = np.std(site_aod, axis=1)
        a_aod_num[i_t] = np.sum(from_site, axis=1)
        
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
    lon_diffs = np.abs(lons[:,np.newaxis] - df_m.longitudes)
    site_lon_idx = np.argmin(lon_diffs, axis=1)
    lat_diffs = np.abs(lats[:,np.newaxis] - df_m.latitudes)
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
    
    # Get the latitudes, longitudes, and aod data
    m_lons = df_m.longitudes[m_lon_idx]
    m_lats = df_m.latitudes[m_lat_idx]
    m_ll = zip(m_lons, m_lats)
    m_aod = df_m.aod_d[:, m_lat_idx, m_lon_idx]
        
    # Now find the nearest neighbours and average
    m_aod_avg = np.zeros_like(a_aod_avg)
    m_aod_std = np.zeros_like(a_aod_std)
    m_aod_num = np.zeros_like(a_aod_num)
    
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
    
    # REARANGE THE OUTPUT TO BE IN THE REQUIRED FORMAT
    # Reshape the longitudes, latitudes, and times to be the shape of the grid
    [i_1, i_2] = np.indices(a_aod_avg.shape)
    times = times[i_1]
    lons, lats = lons[i_2], lats[i_2]
    
    # Unravel the arrays
    avg = np.array([m_aod_avg.ravel(), a_aod_avg.ravel()])
    std = np.array([m_aod_std.ravel(), a_aod_std.ravel()])
    num = np.array([m_aod_num.ravel(), a_aod_num.ravel()])
    times = times.ravel()
    lons, lats = lons.ravel(), lats.ravel()
    
    # Return only elements for which there is both satellite and AERONET data
    r = (num[0] > 0) & (num[1] > 0)
    return [avg[:,r], std[:,r], num[:,r], lons[r], lats[r], times[r]]


def model_sat_match(df_m, df_s, match_time, match_dist):
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
        (Only accurate for less than ~2.5)
    '''
    
    if (type(df_m.aod_d) == type(None)) | (type(df_s.aod_d) == type(None)):
        raise ValueError('Both data frames must have dust AOD data.')
    
    K = 10  # Number of nearest neighbours to find for each location
    
    # Include only dust AOD data for MODIS
    if df_s.data_set == 'modis':
        is_dust = df_s.aod_d[1]
        s_times = df_s.times[is_dust]
        s_lons = df_s.longitudes[is_dust]
        s_lats = df_s.latitudes[is_dust]
        s_aod_d = df_s.aod_d[0]
    
    # Bin the time
    t_mult = 60 / match_time
    m_times = np.rint(df_m.times * t_mult) / t_mult
    s_times = np.rint(s_times * t_mult) / t_mult
    
    # Get the list of times included in both sets of data
    times = []
    for t in np.unique(m_times):
        if np.any(s_times == t):
            times.append(t)
    times = np.array(times)
    
    # Firstly bin the data into a grid of latitude and longitude
    s_lons = np.rint(s_lons / match_dist) * match_dist
    s_lats = np.rint(s_lats / match_dist) * match_dist
    s_ll = np.array([s_lons, s_lats])
    m_lons = np.rint(df_m.longitudes / match_dist) * match_dist
    m_lats = np.rint(df_m.latitudes / match_dist) * match_dist
    
    lons, lats, times_arr = [], []
    s_aod_avg, s_aod_std = [], []
    
    for i_t, time in enumerate(times):
        s_ll_t = s_ll[:, s_times==time]
        s_aod_t = s_aod_d[s_times==time]
        
        # FIND THE MEAN AND STANDARD DEVIATION OF THE SATELLITE DATA IN EACH GRID CELL
        # Sort the latitudes and longitudes to bring duplicate locations together
        sort_idx = np.lexsort(s_ll_t)
        sorted_ll = s_ll_t[:, sort_idx]
        sorted_aod = s_aod_t[sort_idx]
        
        # Obtain a mask to indicate where the new locations begin
        uniq_mask = np.append(True, np.any(np.diff(sorted_ll, axis=1), axis=0))
        
        # Find an ID number to identify the location of each member of the sorted arrays
        ID = np.cumsum(uniq_mask) - 1
        
        # Take the average and standard deviations for each location
        s_aod_avg_t = np.bincount(ID, sorted_aod) / np.bincount(ID)
        s_aod_std_t = np.bincount(ID, sorted_aod**2) / np.bincount(ID) - s_aod_avg_t**2
        
        lons_t = sorted_ll[0, uniq_mask]
        lats_t = sorted_ll[1, uniq_mask]
        
        # NOW FIND THE MEAN AND STD OF THE UM DATA IN EACH GRID CELL
        # Start by obtaining the location indices of the lats and lons
        loc_idx_mat = np.indices(lons_t.shape).repeat(m_lons.size, axis=0)
        m_lons_loc = loc_idx_mat[m_lons[:,np.newaxis] == lons_t]
        m_lats_loc = loc_idx_mat[m_lats[:,np.newaxis] == lats_t]
        
        # get a list of the location indices and aod data for every matching lat & lon
        m_
        
        
        times_arr.extend(np.full_like(s_aod_avg_t, time))
        s_aod_avg.extend(s_aod_avg_t)
        s_aod_std.extend(s_aod_std_t)
    
    lons, lats, times_arr = np.array(lons), np.array(lats), np.array(times_arr)
    s_aod_avg, s_aod_std = np.array(s_aod_avg), np.array(s_aod_std)
    
    
    # NOW GET NEAREST NEIGBOURS TO EACH SATELLITE DATA POINT AT EACH TIME AND AVERAGE
    lons, lats, time_arr = [], [], []
    s_avg, s_std, s_num = [], [], []
    m_avg, m_std, m_num = [], [], []
    
    for i_t, t in enumerate(times):
        at_t = (s_time == t)
        s_lons, s_lats = s_longitudes[at_t], s_latitudes[at_t]
        s_ll = zip(s_lons, s_lats)
        
        lons.extend(s_lons)
        lats.extend(s_lats)
        time_arr.extend(np.full_like(s_lons, t))
        s_avg.extend(s_aod_d[at_t])
        s_std.extend(np.zeros_like(s_lons))
        s_num.extend(np.ones_like(s_lons))
        
        # Select only model data with the same latitude and longitude bounds as sat data
        # and get a 1Dx2 list of longitudes and latitudes to pass to getnn()
        lon_bounds = [np.min(s_lons), np.max(s_lons)]
        lat_bounds = [np.min(s_lats), np.max(s_lats)]
        
        m_lons, m_lats = df_m.longitudes, df_m.latitudes
        m_lons_idx = np.nonzero((m_lons >= lon_bounds[0]) & (m_lons <= lon_bounds[1]))[0]
        m_lats_idx = np.nonzero((m_lats >= lat_bounds[0]) & (m_lats <= lat_bounds[1]))[0]
        m_lons = m_lons[m_lons_idx]
        m_lats = m_lats[m_lats_idx]
        
        m_aod_t = df_m.aod_d[i_t, m_lats_idx[:, np.newaxis], m_lons_idx].ravel()
        m_lons = np.repeat(m_lons[np.newaxis, :], len(m_lats_idx), axis=0).ravel()
        m_lats = np.repeat(m_lats[:, np.newaxis], len(m_lons_idx), axis=1).ravel()
        m_ll = zip(m_lons, m_lats)
        
        # Add nan to prevent out of array references from getnn()
        m_aod_t = np.append(m_aod_t, np.nan)
        
        # For each site find the indices of the nearest 10 satellite points within
        # match_rad using cKDTree.
        m_nn_idx, dist = getnn(m_ll, s_ll, match_rad, k=K)
        
        # Suppress warnings from averaging over empty arrays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # Average satellite data
            m_avg.extend(np.nanmean(m_aod_t[m_nn_idx], axis=1))
            m_std.extend(np.nanstd(m_aod_t[m_nn_idx], axis=1))
            m_num.extend(np.sum(np.isfinite(dist), axis=1))
    
    aod = np.array([m_avg, s_avg])
    std = np.array([m_std, s_std])
    num = np.array([m_num, s_num])
    lons, lats, times = np.array(lons), np.array(lats), np.array(time_arr)
    
    # Return only elements for which there is both model and satellite data
    r = (num[0] > 0) & (num[1] > 0)
    return [aod[:,r], std[:,r], num[:,r], lons[r], lats[r], times[r]]


def collocate(df1, df2, match_time=30, match_rad=25):
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
    if (df1.cube == None) & (df2.cube == None):
        
        if df2.data_set == 'aeronet':
            # params  has the form [aod, std, num, lon, lat, time]
            params = sat_anet_match(df1, df2, match_time, match_rad)
            
        elif df1.data_set == 'aeronet':
            params = sat_anet_match(df2, df1, match_time, match_rad)
            param012 = [params[i][::-1] for i in range(3)]
            params = param012 + params[3:]
        
        [aod, std, num, lon, lat, time] = params
        
        return MatchFrame(aod, std, num, lon, lat, time, df1.date, match_time, match_rad,
                          df1.wavelength, forecasts, data_sets, aod_type=0)
    
    # Model-AERONET match-up
    elif (df1.cube != None) & (df2.data_set == 'aeronet'):
        params = model_anet_match(df1, df2, match_time, match_rad)
        
        [aod, std, num, lon, lat, time] = params
        
        return MatchFrame(aod, std, num, lon, lat, time, df1.date, match_time, match_rad,
                          df1.wavelength, forecasts, data_sets, aod_type=1)
    
    # Same as above but the other way around
    elif (df1.data_set == 'aeronet') & (df2.cube != None):
        params = model_anet_match(df2, df1, match_time, match_rad)
        param012 = [params[i][::-1] for i in range(3)]
        params = param012 + params[3:]
        
        [aod, std, num, lon, lat, time] = params
        
        return MatchFrame(aod, std, num, lon, lat, time, df1.date, match_time, match_rad,
                          df1.wavelength, forecasts, data_sets, aod_type=1)
    
    # Model-Satellite match-up
    elif (df1.cube != None) & (df2.cube == None):
        params = model_sat_match(df1, df2, match_time, match_rad)
        
        [aod, std, num, lon, lat, time] = params
        
        return MatchFrame(aod, std, num, lon, lat, time, df1.date, match_time, match_rad,
                          df1.wavelength, forecasts, data_sets, aod_type=1)
    
    # Same as above but the other way around
    elif (df1.cube == None) & (df2.cube != None):
        params = model_sat_match(df2, df1, match_time, match_rad)
        param012 = [params[i][::-1] for i in range(3)]
        params = param012 + params[3:]
        
        [aod, std, num, lon, lat, time] = params
        
        return MatchFrame(aod, std, num, lon, lat, time, df1.date, match_time, match_rad,
                          df1.wavelength, forecasts, data_sets, aod_type=1)
    
    # Model-Model match-up
    elif (df1.cube != None) & (df2.cube != None):
        
        if (type(df1.aod_d) == type(None)) | (type(df2.aod_d) == type(None)):
            raise ValueError('Both data frames must have dust AOD data.')
        
        # Get times shared between the two data frames
        in_shared_times = np.array([time in df2.times for time in df1.times])
        times = df1.times[in_shared_times]
        times1_idx = np.arange(len(df1.times))[in_shared_times]
        
        aod = np.zeros((2, len(times)) + df1.aod_d[0].shape)
        cube_data = np.zeros((len(times),) + df1.aod_d[0].shape)
        for i_t, time in enumerate(times):
            aod[0, i_t] = df1.aod_d[df1.times==time][0]
            aod[1, i_t] = df2.aod_d[df2.times==time][0]
            cube_data[i_t] = aod[1, i_t] - aod[0, i_t]
            
        # Get data to put into MatchFrame
        std = np.zeros_like(aod)        # No averaging
        num = np.ones_like(aod)         # is performed
        lon, lat = np.meshgrid(df1.longitudes, df1.latitudes)
        lon_f = np.tile(lon.ravel(), len(times))            # Same dimensions as
        lat_f = np.tile(lat.ravel(), len(times))            # flattened data
        cube = df1.cube[times1_idx]
        cube.data = cube_data   # Cube with data of df2 - df1
        
        return MatchFrame(aod, std, num, lon_f, lat_f, times, df1.date, None, None,
                          df1.wavelength, forecasts, data_sets, aod_type=1, cube=cube)



if __name__ == '__main__':
    Lon1 = np.random.random(2560)
    Lat1 = np.random.random(2560)
    Lon2 = np.random.random(206)
    Lat2 = np.random.random(206)
    d1 = np.asarray(zip(Lon1, Lat1))
    d2 = np.asarray(zip(Lon2, Lat2))
    tic = time.time()
    i, d = getnn(d1, d2, 0.1, k=2)
    toc = time.time()
    print(toc-tic)
#     print(i)
#     print(d)
#     print(d1[i])