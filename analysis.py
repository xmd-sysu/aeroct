'''
This module contains functions to obtain a list of MatchFrame objects given a list of
dates. It also contains the functions to plot the data over time.

Created on Jul 3, 2018

@author: savis
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
sys.path.append('/home/h01/savis/workspace/summer')
import aeroct

scratch_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/'


def datetime_list(initial_date, days):
    '''
    Return a list of datetimes (at 00:00:00) beginning at initial_date. The days argument
    is a list for which each element gives the number of days after initial date for each
    element of the returned list.
    Eg. datetime_list(initial_date='20180624', days=[-3,4,7]) will return datetime a list
    of datetime objects of: [2018-06-21, 2018-06-28, 2018-07-02]
    
    Parameters:
    initial_date : str
        The date corresponding to days=0. Format is 'YYYYMMDD'.
    days : integer array
        The days after initial_date to return datetime objects.
    '''
    initial_date = datetime.strptime(initial_date, '%Y%m%d')
    dt_list = [initial_date + timedelta(days=d) for d in days]
    return dt_list


def period_download_and_match(data_set1, data_set2, dates, data_set3=None,
                              forecast_time=(0, 0, 0), match_time=30, match_rad=25,
                              save=True, dir_path=scratch_path+'data_frames/'):
    '''
    Download data for data_set1 and data_set2 for the given dates. These are then
    collocated and the resulting MatchFrames for each date are returned in a list. By
    default the MatchFames are saved individually as they are created, this may be
    changed using the 'save' parameter.
    If data_set3 is provided then all three data sets are downloaded and matched up
    together. In this case a list for each of the three combinations is returned. The
    order is then 1-2, 1-3, 2-3.
    
    Parameters:
    data_set1, data_set2 : str
        These provide the names of the data sets to download and match. The possible
        values are: 'aeronet', 'modis', and 'metum' (Unified model).
        When plotted on a scatter plot data_set1 is put on the x-axis and data_set2 on
        the y-axis.
    dates : datetime list
        The dates for which to match up. This may be easily obtained using the
        'datetime_list' function.
    data_set3 : str, optional (Default: None)
        This is similar to data_set1/2 however if supplied then a three-way match-up is
        performed.
    forecast_time : iterable of ints (Default: (0, 0))
        The forecast lead time to use. If metum is chosen as a data set then the
        corresponding value in the tuple must be filled.
    match_time : float, optional (Default: 30 (minutes))
        The time over which data will be matched and averaged in minutes.
    match_rad : int, optional (Default: 25 (km))
        The radius for which data will be matched and averaged in kilometers.
    save : bool (Default: True)
        Choose whether to save the individual MatchFrames using MatchFrame.dump().
    dir_path : str (Default: '/scratch/{USER}/aeroct/data_frames')
        The path to the directory where the files will be saved if save=True.
    '''
    
    match_df12s = []
    match_df13s = []
    match_df23s = []
    
    for date in dates:
        date = date.strftime('%Y%m%d')
        
        df1 = aeroct.load(data_set1, date, forecast_time=forecast_time[0])
        df2 = aeroct.load(data_set2, date, forecast_time=forecast_time[1])
        
        match_df12 = aeroct.collocate(df1, df2, match_time, match_rad)
        if save == True:
            match_df12.dump(dir_path=dir_path)
        
        if data_set3 == None:
            match_df12s.append(match_df12)
        else:
            df3 = aeroct.load(data_set3, date, forecast_time=forecast_time[2])
            match_df13 = aeroct.collocate(df1, df3, match_time, match_rad)
            match_df23 = aeroct.collocate(df2, df3, match_time, match_rad)
            match_df13s.append(match_df13)
            match_df23s.append(match_df23)
            
            if save == True:
                match_df13.dump(dir_path = dir_path)
                match_df23.dump(dir_path = dir_path)
    
    if data_set3 == None:    
        return match_df12s
    else:
        return [match_df12s, match_df13s, match_df23s]


def concatenate_period(df_list):
    '''
    Concatenate a list of data frames over a period of time so that the average may be
    plotted on a map. A data frame of the input type (DataFrame or MatchFrame) is
    returned with a date attribute containing the list of dates
    
    Parameters:
    df_list : iterable of DataFrames / MatchFrames
        The list of data frames over a period. All must have the same wavelength and
        data-set(s). 
    '''
    # Currently only works for MatchFrames
    if df_list[0].__class__.__name__ == 'MatchFrame':
        
        match_time = df_list[0].match_time
        match_rad = df_list[0].match_radius
        wavelength = df_list[0].wavelength
        fc_times = df_list[0].forecast_times
        data_sets = df_list[0].data_sets
        aod_type = df_list[0].aod_type
        
        dates = []
        data, data_std, data_num = [], [], []
        longitudes, latitudes, times = [], [], []
        
        for df in df_list:
            
            # Check that the wavelengths and data-sets all match
            if df.wavelength != df_list[0].wavelength:
                raise ValueError('The list of data frames do not contain data for the same\
                                  wavelength.')
            if df.data_sets != df_list[0].data_sets:
                raise ValueError('The list of data frames do not contain data from the\
                                  same data-sets.')
            
            dates.append(df.date)
            data.extend(df.data)
            data_std.extend(df.data)
            data_num.extend(df.data_num)
            longitudes.extend(df.longitudes)
            latitudes.extend(df.latitudes)
            times.extend(times)
        
        data, data_std, data_num = np.array(data), np.array(data_std), np.array(data_num)
        longitudes, latitudes = np.array(longitudes), np.array(latitudes)
        times = np.array(times)
        
        return aeroct.MatchFrame(data, data_std, data_num, longitudes, latitudes, times,
                                 dates, match_time, match_rad, wavelength, fc_times,
                                 data_sets, aod_type)


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


if __name__ == '__main__':
    data_set1 = 'aeronet'
    data_set2 = 'modis'
    dates = datetime_list('20180620', np.arange(0, 15, 5))
    match_dfs = period_download_and_match(data_set1, data_set2, dates)
    
    Rs = [match_dfs[i].R for i in range(len(match_dfs))]
    plt.plot(dates, Rs)
    plt.show()