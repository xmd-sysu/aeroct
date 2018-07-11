'''
This module and processes the data retrieved using the download module so that it may
easily be passed into a data_frame class. This will allow it to be more easily be
compared with other data sources.

Created on Jun 25, 2018

@author: savis
'''

from __future__ import division
from datetime import datetime
import numpy as np

def process_data(aod_array, date):
    '''
    Process the AOD data from a numpy record array into a list that may be passed into a
    data frame so that it may be compared with other data sources.
    The returned aod_d for dust is a 2x1D array, the first array is the AOD, the second
    is its indices in the the full array.
    
    Parameter:
    aod_array : rec array
        The data obtained from the download module.
    date : str or datetime
        The date for which to retrieve records. Format: YYYYMMDD for strings. Do not
        include a time if a datetime is used.
    '''
    
    if type(date) is not datetime:
        date = datetime.strptime(date, '%Y%m%d')
    
    not_mask = aod_array['AOD_NM550'] > -0.05
    is_dust = aod_array['ARSL_TYPE'] == 1
    condition = not_mask
    condition_d = not_mask & is_dust
    
    # Find the indices of the dust AODs within the total AOD array
    condition_dust_idx = aod_array['ARSL_TYPE'][condition] == 1
    dust_idx = np.indices([aod_array['ARSL_TYPE'][condition].size])[0, condition_dust_idx]
    
    aod = aod_array['AOD_NM550'][condition]                     # All AODs
    aod_d = [aod_array['AOD_NM550'][condition_d], dust_idx]     # Only dust AODs
    lat = aod_array['LTTD'][condition]
    lon = aod_array['LNGD'][condition]
    time = aod_array['TIME'][condition]                     # Hours since 00:00:00
    wl = 550    # wavelength [nm]
    
    return [aod, aod_d, lat, lon, time, date, wl]