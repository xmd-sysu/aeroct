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

def process_data(aod_array, date, satellite='Both', src=None):
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
    satellite : {'Both', 'Terra, 'Aqua'}, optional (Default: 'Both')
        Which satellite's data to load.
    src : str, optional (Default: None)
        The source from which the data has been retrieved.
        None or 'MetDB' for MetDB extraction (Note: fewer dust filters available)
        'NASA' for download from ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/
    '''
    
    if type(date) is not datetime:
        date = datetime.strptime(date, '%Y%m%d')
    
    # Select only the unmasked data from the chosen satellite(s)
    if satellite == 'Terra':
        chosen_sat = aod_array['STLT_IDNY'] == 783
    elif satellite == 'Aqua':
        chosen_sat = aod_array['STLT_IDNY'] == 784
    else:
        chosen_sat = True
    not_mask = aod_array['AOD_NM550'] > -0.05001
    condition = chosen_sat & not_mask
    aod_array = aod_array[condition]
    
    aod = aod_array['AOD_NM550']   # Total AOD
    lat = aod_array['LTTD']
    lon = aod_array['LNGD']
    time = aod_array['TIME']       # Hours since 00:00:00
    wl = 550    # wavelength [nm]
    
    # Find the elements of the AOD data which satisfy various dust filter conditions
    # Then store these filters in a dictionary
    # Also remove data which is masked (-9999.0)
    if (src is None) | (src == 'MetDB'):
        filter_type_land = (aod_array['ARSL_TYPE'] == 1)
        # No other filters possible
        filter_ae_land = np.full_like(filter_type_land, True)
        filter_ssa_land = np.full_like(filter_type_land, True)
        filter_fm_frc_ocean = np.full_like(filter_type_land, True)
        filter_ae_ocean = np.full_like(filter_type_land, True)
        filter_region_mask_ocean = np.full_like(filter_type_land, True)
    
    elif src == 'NASA':
        filter_type_land = (aod_array['ARSL_TYPE'] == 5)
        filter_ae_land = (aod_array['AE_LAND'] > - 0.1) & (aod_array['AE_LAND'] <= 0.6)
        filter_ssa_land = (0.878 < aod_array['SSA_LAND']) & (aod_array['SSA_LAND'] < 0.955)
        filter_fm_frc_ocean = (aod_array['FM_FRC_OCEAN'] >= 0) & (aod_array['FM_FRC_OCEAN'] <= 0.45)
        filter_ae_ocean = (aod_array['AE_OCEAN'] > - 0.1) & (aod_array['AE_OCEAN'] <= 0.6)
        
        filter_region_mask_ocean = (aod_array['FM_FRC_OCEAN'] >= 0) & \
                    (((aod_array['LNGD'] >= -90) & (aod_array['LNGD'] <= 78) &
                      (aod_array['LTTD'] >= 5) & (aod_array['LTTD'] <= 30)) | \
                     ((aod_array['LNGD'] >= 115) & (aod_array['LNGD'] <= 145) &
                      (aod_array['LTTD'] >= 20) & (aod_array['LTTD'] <= 40)))
    
    
    dust_filter = {'ARSL_TYPE_LAND' : filter_type_land,
                   'AE_LAND' : filter_ae_land,
                   'SSA_LAND' : filter_ssa_land,
                   'FM_FRC_OCEAN' : filter_fm_frc_ocean,
                   'AE_OCEAN' : filter_ae_ocean,
                   'REGION_OCEAN' : filter_region_mask_ocean,
                   'NONE': np.full_like(filter_type_land, True)}
    
    return [[aod, None], lat, lon, time, date, wl, dust_filter]