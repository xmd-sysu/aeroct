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
    The output is an array containing the new AOD values for both the total AOD and the
    coarse mode.
    
    Parameters:
    aeronet_df: (pandas data frame) The data frame obtained from the download module.
    wavelength: (int) The wavelength at which to obtain the AOD values.
    '''
    
    find_aod = lambda aod1, wl1, wl2, ae: aod1 * (wl2 / wl1) ** (-ae)
    
    if (wavelength >= 450) & (wavelength <= 550):
        # Use the angstrom exponent given for 500nm
        aod_t1 = np.array(aeronet_df['Total_AOD_500nm[tau_a]'])
        aod_c1 = np.array(aeronet_df['Coarse_Mode_AOD_500nm[tau_c]'])
        alpha = np.array(aeronet_df['Angstrom_Exponent(AE)-Total_500nm[alpha]'])
        alpha_f = np.array(aeronet_df['AE-Fine_Mode_500nm[alpha_f]'])
        fmf = np.array(aeronet_df['FineModeFraction_500nm[eta]'])
        alpha_c = np.divide((alpha - alpha_f*fmf), (1 - fmf),
                    out=np.zeros_like(alpha), where=(fmf!=1))
        
        wl1 = 500
        aod_t2 = find_aod(aod_t1, wl1, wavelength, alpha)
        aod_c2 = find_aod(aod_c1, wl1, wavelength, alpha_c)
    
    else:
        raise ValueError('Wavelength ({} nm) out of range.'.format(wavelength))
    
    return [aod_t2, aod_c2]


def process_data(aeronet_df, date, wavelength=550):
    '''
    Process the AOD data from the pandas data frame into a list that may be passed into
    the AeroCT data frame so that it may be compared with other data sources. The aerosol
    optical depth is evaluated at the given wavelength using the angstrom exponent.
    
    Parameter:
    aeronet_df : pandas data frame
        The data frame obtained from the download module.
    date: str
        The date for which the data has been downloaded. Format is 'YYYYMMDD'.
    wavelength: int, optional (Default: 550 (nm))
        The wavelength at which to obtain the AOD values.
    '''
    date = datetime.strptime(date, '%Y%m%d')
    
    # Drop rows with nan values
    if (wavelength >= 450) & (wavelength <= 550):
        aeronet_df = aeronet_df[np.isfinite(aeronet_df['Total_AOD_500nm[tau_a]'])]
    
    aod = interpolate_aod(aeronet_df, wavelength) # AOD data in form: [Total, Coarse-mode]
    lat = np.array(aeronet_df['Site_Latitude(Degrees)'])
    lon = np.array(aeronet_df['Site_Longitude(Degrees)'])
    sites = np.array(aeronet_df['AERONET_Site'])
    
    # Get hours since 00:00:00
    total_hours = lambda td: td.seconds / 3600 + td.days * 24
    time = np.array([total_hours(dt - date) for dt in aeronet_df['datetime']])
    
    return [aod, lon, lat, time, date, wavelength, sites]


if __name__ == '__main__':
    pass