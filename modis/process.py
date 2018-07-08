'''
This module and processes the data retrieved using the download module so that it may
easily be passed into a data_frame class. This will allow it to be more easily be
compared with other data sources.

Created on Jun 25, 2018

@author: savis
'''

from __future__ import division
from datetime import datetime

def process_data(aod_array, date, coarse_mode=False):
    '''
    Process the AOD data from a numpy record array into a list that may be passed into a
    data frame so that it may be compared with other data sources.
    
    Parameter:
    aod_array : rec array
        The data obtained from the download module.
    date : str or datetime
        The date for which to retrieve records. Format: YYYYMMDD for strings. Do not
        include a time if a datetime is used.
    coarse_mode : bool, optional (Default: False)
        False: the total AOD is returned.
        True: the coarse mode AOD is returned (for comparison with UM).
    '''
    
    if type(coarse_mode) != bool:
        raise ValueError('Unrecognised value for coarse_mode: {}'.format(coarse_mode))
    
    if type(date) is not datetime:
        date = datetime.strptime(date, '%Y%m%d')
    
    not_mask = aod_array['AOD_NM550'] > 1e-5
    is_dust = aod_array['ARSL_TYPE'] == 1
    if coarse_mode == False:
        condition = not_mask
    else:
        condition = not_mask & is_dust
    
    aod = aod_array['AOD_NM550'][condition]
    lat = aod_array['LTTD'][condition]
    lon = aod_array['LNGD'][condition]
    wl = 550    # wavelength [nm]
    
    # NOTE: WHAT HAPPENS AT THE BEGINNING OF MONTHS / BEGINNING OF YEARS
    time = (aod_array['DAY'][condition] - date.day) * 24 + (aod_array['HOUR'][condition] - date.hour) + \
           (aod_array['MINT'][condition] - date.minute) / 60   # Hours since 00:00:00
    
    if coarse_mode == False:
        return [aod, None, lat, lon, time, date, wl]
    else:
        return [None, aod, lat, lon, time, date, wl]