'''
Created on Jun 20, 2018

@author: savis
'''
from __future__ import division
import os
from datetime import datetime, timedelta
import pandas as pd
from pandas.compat import StringIO

import matplotlib.pyplot as plt
import numpy as np

def download_data(site=None, date1=None, date2=None, h1=0, h2=23, prd='SDA15', avg=10):
    '''
    Downloads AERONET v3 data and outputs the resulting string
    
    Parameters:
    site: (Optional) (str) Name of the aeronet site, or None for all sites.
        (Default: None)
    date1, date2: (Optional) (str) Dates to begin and end the data collection.
        Format is 'yyyymmdd'.
        (Default: today's date)
    h1, h2: (Optional) (int) Hours for the beginning and end of the data collection.
        May be 0 - 23.
        (Default: 0, 23)
    prd: (Optional) (str) The product code for the data.
        (AOD10, AOD15, AOD20, SDA10, SDA15, SDA20, TOT10, TOT15, TOT20)
        (Default: 'SDA15')
    avg: (Optional) (int) avg=10 for all points, avg=20 for daily average.
        (Default: 10)
    '''
    
    data_host = 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3'
    
    if site != None:
        site_txt = '&site={}'.format(site)
    else:
        site_txt = ''
    
    if date1 == None:
        date1_dt = datetime.utcnow()
    else:
        date1_dt = datetime.strptime(date1, '%Y%m%d')
    y1 = date1_dt.year
    m1 = date1_dt.month
    d1 = date1_dt.day
    
    if date2 == None:
        date2_dt = datetime.utcnow()
    else:
        date2_dt = datetime.strptime(date2, '%Y%m%d')
    y2 = date2_dt.year
    m2 = date2_dt.month
    d2 = date2_dt.day
    
    # Command to download the data, then the output is piped to a string.
    cmd = 'curl -s -k "{}?{}year={}&month={}&day={}&year2={}&' + \
            'month2={}&day2={}&hour={}&hour2={}&{}=1&AVG={}&if_no_html=1"'
    cmd = cmd.format(data_host, site_txt, y1, m1, d1, y2, \
                     m2, d2, str(h1), str(h2), prd, str(avg))
    
    aeronet_data_string = os.popen(cmd).read()
    return aeronet_data_string


def download_data_day(date, site=None, prd='SDA15', avg=10, minutes_err=30):
    '''
    Downloads AERONET v3 data for the given day with some extra data before and after.
    The output is a string containing the data.
    
    Parameters:
    date: (str) The date for which to collect data. Format is 'yyyymmdd'.
    site: (Optional) (str) Name of the aeronet site, or None for all sites.
        (Default: None)
    prd: (Optional) (str) The product code for the data.
        (AOD10, AOD15, AOD20, SDA10, SDA15, SDA20, TOT10, TOT15, TOT20)
        (Default: 'SDA15')
    avg: (Optional) (int) avg=10 for all points, avg=20 for daily average.
        (Default: 10)
    minutes_err: (int, optional) Number of minutes of data to include from the days
        before and after. Default: 30 (min)
    '''
    # Get the start and end dates in format 'YYYYMMDD'
    # and the start and end times from minutes_err
    date1 = (datetime.strptime(date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
    date2 = (datetime.strptime(date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
    h1 = 23 - int(minutes_err / 60)
    h2 = 1 + int(minutes_err / 60)
    
    aeronet_data_string = download_data(date1=date1, date2=date2, h1=h1, h2=h2, prd=prd,
                                        avg=avg)
    return aeronet_data_string


def parse_data(aeronet_data_string, prd='SDA15'):
    '''
    Parse the string obtained in the download_data function into a pandas dataframe.
    
    Parameter:
    aeronet_data_string: (str) The string obtained by the download_data function.
    prd: (Optional) (str) The product code for the data.
        (AOD10, AOD15, AOD20, SDA10, SDA15, SDA20, TOT10, TOT15, TOT20)
        (Default: 'SDA15')
    '''
    
    date_parse = lambda d: pd.datetime.strptime(d, "%d:%m:%Y %H:%M:%S")
    
    hd = pd.read_csv(StringIO(aeronet_data_string), skiprows=5, nrows=1).columns.tolist()
    # update header with dummy wavelength columns
    hd[-1] = 'w1'
    hd.extend(['w' + x for x in map(str, range(2, 11))])
    
    
    aeronet_df = pd.read_csv(StringIO(aeronet_data_string), names=hd, skiprows=6,
                             na_values=[-999.0], parse_dates={'datetime': [1,2]},
                             date_parser=date_parse)
    return aeronet_df


if __name__ == '__main__':
    data_string = download_data_day(date='20180623')
#     with open('AERONET_data', 'w') as text_file:
#         text_file.write(data_string)
    df = parse_data(data_string)
    print(df.iloc[0])
    
    i = 50
    aod = [df['870nm_Input_AOD'][i], df['675nm_Input_AOD'][i], df['500nm_Input_AOD'][i], df['440nm_Input_AOD'][i], df['380nm_Input_AOD'][i]]
    wl = [df['w1'][i], df['w2'][i], df['w3'][i], df['w4'][i], df['w5'][i]]
    plt.plot(wl, aod, 'ro')
    
    ae = df['Angstrom_Exponent(AE)-Total_500nm[alpha]'][i]
    w = np.linspace(wl[0], wl[4], 100)
    aod_ae = df['Total_AOD_500nm[tau_a]'][i] * (w/wl[2]) ** (-ae)
    
    plt.plot(w, aod_ae, 'b--')
    
    plt.show()