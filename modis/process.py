'''
This module and processes the data retrieved using the download module so that it may
easily be passed into a data_frame class. This will allow it to be more easily be
compared with other data sources.

Created on Jun 25, 2018

@author: savis
'''

from __future__ import division
from datetime import datetime, timedelta

def process_data(aod_array, date):
    '''
    Process the AOD data from a numpy record array into a list that may be passed into a
    data frame so that it may be compared with other data sources.
    
    Parameter:
    aod_array: (rec array) The data obtained from the download module.
    date: (str or datetime) The date for which to retrieve records. Format: YYYYMMDD for
        strings. Do not include a time if a datetime is used.
    '''
    
    if type(date) is not datetime:
        date = datetime.strptime(date, '%Y%m%d')
    
    aod = aod_array['AOD_NM550']
    lat = aod_array['LTTD']
    lon = aod_array['LNGD']
    wl = 550    # wavelength [nm]
    
    time = (aod_array['DAY'] - date.day) * 24 + (aod_array['HOUR'] - date.hour) + \
           (aod_array['MINT'] - date.minute) / 60   # Hours since 00:00:00
    
    return [aod, lat, lon, time, date, wl]