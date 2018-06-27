'''
This module processes the pandas data frame obtained from the download module into an
data_frame class so that it may be compared with other data sources. The wavelength of
the AOD data is changed to the desired value by using the angstrom coefficient.

Created on Jun 22, 2018

@author: savis

TODO: Allow products other than SDA15 to be used.
TODO: Find the angstrom exponent from the AOD data 
'''

from __future__ import division
from datetime import datetime
import numpy as np


def interpolate_aod(aeronet_df, wavelength):
    '''
    Find the aerosol optical depth at the given wavelength by obtaining the angstrom
    exponent from the values at the two surrounding wavelengths.
    The output is an array containing the new AOD values.
    
    Parameters:
    aeronet_df: (pandas data frame) The data frame obtained from the download module.
    wavelength: (int) The wavelength at which to obtain the AOD values.
    '''
    
    find_aod = lambda aod1, wl1, wl2, ae: aod1 * (wl2 / wl1) ** (-ae)
    
    if (wavelength >= 450) & (wavelength <= 550):
        # Use the angstrom exponent given for 
        aod1 = np.array(aeronet_df['Total_AOD_500nm[tau_a]'])
        angstrom_exps = np.array(aeronet_df['Angstrom_Exponent(AE)-Total_500nm[alpha]'])
        wl1 = 500
        new_aod = find_aod(aod1, wl1, wavelength, angstrom_exps)
    
    else:
        raise ValueError, 'Wavelength ({} nm) out of range.'.format(wavelength)
    
    return new_aod


def process_data(aeronet_df, date, wavelength=550):
    '''
    Process the AOD data from the pandas data frame into a list that may be passed into
    the AeroCT data frame so that it may be compared with other data sources. The aerosol
    optical depth is evaluated at the given wavelength using the angstrom exponent.
    
    Parameter:
    aeronet_df: (pandas data frame) The data frame obtained from the download module.
    date: (str) The date for which the data has been downloaded. Format is 'YYYYMMDD'.
    wavelength: (int, optional) The wavelength at which to obtain the AOD values.
        Default: 550 (nm)
    '''
    date = datetime.strptime(date, '%Y%m%d')
    
    aod = interpolate_aod(aeronet_df, wavelength)
    lat = np.array(aeronet_df['Site_Latitude(Degrees)'])
    lon = np.array(aeronet_df['Site_Longitude(Degrees)'])
    
    # Get hours since 00:00:00
    total_hours = lambda td: td.seconds / 3600 + td.days * 24
    time = np.array([total_hours(dt - date) for dt in aeronet_df['datetime']])
    
    return [aod, lat, lon, time, date, wavelength]


if __name__ == '__main__':
    pass