'''
This module and processes the data retrieved using the download module so that it may
easily be passed into a data_frame class. This will allow it to be more easily be
compared with other data sources.

Created on Jun 25, 2018

@author: savis
'''

from __future__ import division
from datetime import datetime, timedelta
import numpy as np

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
    
    not_mask = aod_array['AOD_NM550'] > 1e-5
    aod = aod_array['AOD_NM550'][not_mask]
    lat = aod_array['LTTD'][not_mask]
    lon = aod_array['LNGD'][not_mask]
    wl = 550    # wavelength [nm]
    
    time = (aod_array['DAY'][not_mask] - date.day) * 24 + (aod_array['HOUR'][not_mask] - date.hour) + \
           (aod_array['MINT'][not_mask] - date.minute) / 60   # Hours since 00:00:00
    
    return [aod, lat, lon, time, date, wl]