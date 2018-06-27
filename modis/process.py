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
    '''
    
    aod = aod_array['AOD_NM550']
    lat = aod_array['LTTD']
    lon = aod_array['LNGD']
    wl = 550    # wavelength [nm]
    
    total_hours = lambda td: td.seconds / 3600 + td.days * 24
    
    dt = datetime(aod_array['YEAR'], aod_array['MNTH'], aod_array['DAY'],
                  aod_array['HOUR'], aod_array['MINT'])
    time = total_hours(dt - date)  # Hours since 00:00:00
    
    return [aod, lat, lon, time, date, wl]