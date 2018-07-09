'''
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


def date_range_download_and_match(data_set1, data_set2, dates, data_set3=None,
                                  forecast_time=(None, None, None), match_time=30,
                                  match_rad=25, save=True, dir_path=scratch_path+'data_frames/'):
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
    forecast_time : int tuple (Default: (0, 0))
        The forecast lead time to use. If metum is chosen as the first or second data set
        then the corresponding value in the tuple must be filled.
    match_time : float, optional (Default: 30 (minutes))
        The time over which data will be matched and averaged in minutes.
    match_rad : int, optional (Default: 25 (km))
        The radius for which data will be matched and averaged in kilometers.
    save : bool (Default: True)
        Choose whether to save the individual MatchFrames using MatchFrame.dump().
    dir_path : str (Default: '/scratch/{USER}/aeroct/data_frames')
        The path to the directory where the files will be saved if save=True.
    '''
    
    match_dfs = []
    
    for date in dates:
        date = date.strftime('%Y%m%d')
        
        df1 = aeroct.load(data_set1, date, forecast_time=forecast_time[0])
        df2 = aeroct.load(data_set2, date, forecast_time=forecast_time[1])
        
        match_df = aeroct.collocate(df1, df2, match_time, match_rad)
        match_dfs.append(match_df)
        
        if save == True:
            match_df.dump(dir_path=dir_path)
    
    return match_dfs


if __name__ == '__main__':
    data_set1 = 'aeronet'
    data_set2 = 'modis'
    dates = datetime_list('20180620', np.arange(0, 15, 5))
    match_dfs = date_range_download_and_match(data_set1, data_set2, dates)
    
    Rs = [match_dfs[i].R for i in range(len(match_dfs))]
    plt.plot(dates, Rs)
    plt.show()