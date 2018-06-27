'''
This module is used to retrieve MODIS AOD data at 550nm from MetDB.

Created on Jun 25, 2018

@author: savis

TODO: Write a method to download the data from the site as an alternative method. Use the
      modis_hdf module and the following code to help with this:
    ftp = ftplib.FTP('ladsftp.nascom.nasa.gov')
    ftp.cwd('/allData/51/MYD04_L2/2016/001')
    data = []
    ftp.dir(data.append)
'''

import os
from metdb import obs
import pwd
from datetime import datetime, timedelta

user = os.getenv('USER')
contact = '{}@metoffice.gov.uk'.format(pwd.getpwnam(user).pw_gecos)
elements = ['YEAR', 'MNTH', 'DAY', 'HOUR', 'MINT', 'LTTD', 'LNGD', 'AOD_NM550']


def retrieve_data_range(start, stop):
    '''
    Retrieve MODIS AOD data at 550nm from MetDB between two times. The output is a
    NumPy record array containing time (YEAR, MNTH, DAY, HOUR, MINT), latitude (LTTD),
    longitude (LNGD), and AOD (AOD_NM550).
    
    Parameters:
    start: (str or datetime) the time from which to begin extracting data. If a string
        is used then it must be in the format YYYYMMDD/HHMM.
    stop: (str or datetime) the time from which to finish extracting data. If a string
        is used then it must be in the format YYYYMMDD/HHMM.
    '''
    
    if type(start) is datetime:
        start = start.strftime('%Y%m%d/%H%MZ')
    if type(stop) is datetime:
        stop = stop.strftime('%Y%m%d/%H%MZ')
    keywords = ['START TIME ' + start, 'END TIME ' + stop]
    
    aod_array = obs(contact, 'SATAOD', keywords, elements, hostname=None)
    return aod_array


def retrieve_data_day(date, minutes_err=30):
    '''
    Retrieve MODIS AOD data at 550nm from MetDB for a single day and some data from the
    day before and after. The output is a NumPy record array containing time (YEAR, MNTH,
    DAY, HOUR, MINT), latitude (LTTD), longitude (LNGD), and AOD (AOD_NM550).
    
    Parameters:
    date: (str or datetime) The date for which to retrieve records. Format: YYYYMMDD for
        strings. Do not include a time if a datetime is used.
    minutes_err: (int, optional) Number of minutes of data to include from the days
        before and after. Default: 30 (min)
    '''
    
    if type(date) is not datetime:
        date = datetime.strptime(date, '%Y%m%d')
    
    start = date - timedelta(minutes=minutes_err)
    stop = date + timedelta(days=1) + timedelta(minutes=minutes_err)
    
    aod_array = retrieve_data_range(start, stop)
    return aod_array


if __name__ == '__main__':
    data = retrieve_data_day(datetime(2018, 06, 23))
    print(data)