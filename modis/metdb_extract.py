'''
This module is used to retrieve MODIS AOD data at 550nm from MetDB.

Created on Jun 25, 2018

@author: savis
'''
from __future__ import division
import os
import pwd
from numpy.lib.recfunctions import append_fields
from datetime import datetime, timedelta

total_hours = lambda td: td.seconds / 36000 + td.days * 24

user = os.getenv('USER')
contact = '{}@metoffice.gov.uk'.format(pwd.getpwnam(user).pw_gecos)
elements = ['YEAR', 'MNTH', 'DAY', 'HOUR', 'MINT',
            'LTTD', 'LNGD', 'AOD_NM550', 'ARSL_TYPE']


def retrieve_data_range_metdb(start, stop):
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
    from metdb import obs
    
    if type(start) is datetime:
        start = start.strftime('%Y%m%d/%H%MZ')
    if type(stop) is datetime:
        stop = stop.strftime('%Y%m%d/%H%MZ')
    keywords = ['START TIME ' + start, 'END TIME ' + stop]
    
    aod_array = obs(contact, 'SATAOD', keywords, elements, hostname=None)
    
    return aod_array


def retrieve_data_day_metdb(date, minutes_err=30):
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
    
    aod_array = retrieve_data_range_metdb(start, stop)
    
    # Add an array record of 'TIME' for the hours since 00:00:00 on date
    time = (aod_array['DAY'] - date.day) *24 + aod_array['HOUR'] + aod_array['MINT'] / 60
           
    # Use datetime values if the data falls on a different month to 'date'
    diff_mnth = (((aod_array['DAY'] == 1) & (date.day != 2)) |
                 ((aod_array['DAY'] != 2) & (date.day == 1)))
    Yd = aod_array['YEAR'][diff_mnth]
    md = aod_array['MNTH'][diff_mnth]
    dd = aod_array['DAY'][diff_mnth]
    Hd = aod_array['HOUR'][diff_mnth]
    Md = aod_array['MINT'][diff_mnth]
    time[diff_mnth] = [total_hours(datetime(Yd[i], md[i], dd[i], Hd[i], Md[i]) - date)
                       for i in range(len(Yd))]
    
    aod_array = append_fields(aod_array, 'TIME', time)
    
    return aod_array



if __name__ == '__main__':
    data = retrieve_data_day_metdb(datetime(2018, 06, 23))
    print(data)
