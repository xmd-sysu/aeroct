'''
Created on Jul 10, 2018

@author: savis
'''
from __future__ import division
import os
import re
import glob
from datetime import datetime
import numpy as np
from urllib2 import urlopen, Request, URLError
from contextlib import closing
from shutil import copyfileobj
try:
    from pyhdf.SD import SD, SDC
    from pyhdf.error import HDF4Error
    h4err = None
except ImportError as h4err:
    pass


def download_hdf_day(date, dl_dir, satellite='Both'):
    src_url='https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/'
    
    # Make dl_dir if the directory does not exist
    if not os.path.exists(dl_dir):
        os.makedirs(dl_dir)
    
    # Get an iterable of the codes for the satellites used in the urls 
    if satellite == 'Both':
        sat_codes = ['MOD04_L2', 'MYD04_L2']
    elif satellite == 'Terra':
        sat_codes = ['MOD04_L2']
    elif satellite == 'Aqua':
        sat_codes = ['MYD04_L2']
    
    # Convert date format from yyyymmdd to yyyy/jjj for use in the url
    date_dt = datetime.strptime(date, '%Y%m%d')
    date_yj = date_dt.strftime('%Y/%j')
    
    num_downloaded = 0
    
    for sat in sat_codes:
        dir_url = src_url + sat + '/' + date_yj + '/'
        
        # Get list of filenames for the date
        filename_pattern = sat + '\.' + '.+?.hdf'
        try:
            dir_r = urlopen(Request(dir_url))
        except URLError, e:
            print(e.reason)
        filenames = re.findall(filename_pattern, dir_r.read())
        
        # Download hdf files
        for filename in filenames:
            if not os.path.exists(dl_dir + filename):
                num_downloaded += 1
                file_url = dir_url + filename
                
                try:
                    with closing(urlopen(file_url)) as file_r:
                        with open(dl_dir + filename, 'wb') as write:
                            copyfileobj(file_r, write)
                except URLError, e:
                    print(e.reason)
        
    if num_downloaded > 0:
        print('Download complete - {} files downloaded to: {}'.format(num_downloaded,
                                                                      dl_dir))
    else:
        print('Files already exist in {}'.format(dl_dir))


def load_data_day(date, dl_dir, download=True, satellite='Both', keep_files=False):
    '''
    This function can be used to download MODIS data for a day. A dictionary is returned
    containing 1D arrays with the following fields:
    'LNGT': longitudes, 'LTTD': latitudes, 'AOD_NM550' : AOD,
    'ARSL_TYPE': aerosol type (dust=1), 'ARSL_RTVL_CNFC_FLAG': quality flag (0
    'YEAR', 'MNTH', 'DAY', 'HOUR', 'MINT': times
    '''
    
    # Download?
    if download == True:
        download_hdf_day(date, dl_dir, satellite)
    
    print('Loading MODIS HDF files.')
    
    # Convert date format from yyyymmdd to yyyyjjj to find the HDF filepaths
    date_dt = datetime.strptime(date, '%Y%m%d')
    date_yj = date_dt.strftime('%Y%j')
    
    files = glob.glob(dl_dir + '*' + date_yj + '.*.hdf')
    
    # Get the fields from the files and concatenate them in lists
    lon, lat, time, arsl_type, aod = [], [], [], [], []
    fieldnames = ['Longitude', 'Latitude', 'Scan_Start_Time', 'Aerosol_Type_Land', 
                   'AOD_550_Dark_Target_Deep_Blue_Combined',
                   'AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag']
    
    for f in files[::-1]:
        parser = h4Parse(f)
        scaled = parser.get_scaled(fieldnames)
        
        # Convert 'Scan_Start_Time' (seconds since 1993-01-01)
        # to hours since 00:00:00 on date
        date_hours = (date_dt - datetime(1993,1,1)).days * 24
        time_hours = scaled['Scan_Start_Time'] / 3600 - date_hours
        
        # Include only the data with the highest quality flag
        highest_qf = (scaled['AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag'] == 3)
        lon.extend(scaled['Longitude'][highest_qf])
        lat.extend(scaled['Latitude'][highest_qf])
        time.extend(time_hours[highest_qf])
        arsl_type.extend(scaled['Aerosol_Type_Land'][highest_qf])
        aod.extend(scaled['AOD_550_Dark_Target_Deep_Blue_Combined'][highest_qf])
    
    fields_dict = {'LNGD' : np.array(lon).ravel(),
                   'LTTD' : np.array(lat).ravel(),
                   'TIME' : np.array(time).ravel(),
                   'ARSL_TYPE' : np.array(arsl_type).ravel(),
                   'AOD_NM550' : np.array(aod).ravel()}
    
    # Remove files?
    if keep_files == False:
        for f in files:
            os.remove(f)
    if not os.listdir(dl_dir):
        os.rmdir(dl_dir)
    
    return fields_dict


class h4Parse(object):
    """
    A pyhdf interface to parse hdf4 file.

    Examples
    --------
    >>> d = h4Parse('file.hdf')
    >>> print d.items  # print available datasets in hdf file

    Author: yaswant.pradhan
    """

    def __init__(self, filename=None):
        # if hdf4import is False:
        if h4err:
            raise ImportError(
                "{}, which is required to read '{}'".format(
                    h4err, os.path.basename(filename)))
        self.sds = ''
        self.items = []
        self.attr = []
        self.filename = filename
        if filename:
            self._populate_SD()
    
    
    def set_filename(self, filename):
        """Set or update hdf filename"""
        self.filename = filename
        self._populate_SD()
    
    
    def _populate_SD(self):
        """Populate SDs and their shape attributes"""

        try:
            h4 = SD(self.filename, mode=SDC.READ)
            self.sds = sorted(h4.datasets().keys())
            self.attr.append(h4.attributes())
            for k, v in sorted(h4.datasets().viewitems()):
                self.items.append((k, v[1]))
            h4.end()
        except HDF4Error as e:
            raise HDF4Error('{}: {}'.format(e, self.filename))
    
    
    def get_sds(self, fieldnames=[]):
        """
        Returns specific or all SDS in the hdf file as dictionary.

        SDS arrays can be accessed using the 'data' key. Note that no scaling
        is applied to the data in get() method (use get_scaled() to achieve
        that). However, the scaling and missing data information can be
        accessed using the following keys:
            'scale_factor'
            'add_offset'
            '_FillValue'
        """
        # Convert scalar fieldnames to list
        if not isinstance(fieldnames, list):
            fieldnames = [fieldnames]
        # Open file to read SDs
        try:
            h4 = SD(self.filename, mode=SDC.READ)
            sclinfo = None
            if 'Slope_and_Offset_Usage' in h4.attributes():
                sclinfo = 'Slope_and_Offset_Usage'
            # Get all available SDS from file if fieldnames in not given
            if len(fieldnames) == 0:
                fieldnames = []
                for key in sorted(h4.datasets()):
                    fieldnames.append(key)
            # Create and empty dataset dictionary with all available
            # fields fill in data from SDS
            sds = dict.fromkeys(fieldnames, {})
            for key in sds:
                attrs = h4.select(key).attributes()
                if sclinfo:
                    attrs[sclinfo] = h4.attributes()[sclinfo]

                sds[key] = attrs
                sds[key]['data'] = h4.select(key).get()
            # Close hdf interface
            h4.end()
        except HDF4Error as e:
            raise HDF4Error(e)

        # Return raw (possibly un-calibrated) SDS/attributes dictionary
        return sds
    
    
    def get_scaled(self, fieldnames=[]):
        """
        Return scaled data assuming that scale_factor and add_offset are
        available in dataset attributes.

        Not a general purpose method, so should be used with caution.
        """
        temp = self.get_sds(fieldnames)
        # print fieldnames
        # print temp[fieldnames].keys()
        # print dir(temp)
        # print temp.keys()
        scaled = dict.fromkeys(temp.keys(), None)
        fillvalue = {}
        for k in scaled:
            # see h4.attributes()['Slope_and_Offset_Usage']
            fillvalue[k] = temp[k]['_FillValue']
            scaled[k] = temp[k]['data'] * (temp[k]['scale_factor']
                                           - temp[k]['add_offset'])

            w = np.where(temp[k]['data'] == fillvalue[k])
            scaled[k][w] = fillvalue[k]

        # Add FillValues information
        scaled['_FillValues'] = fillvalue

        # Return scaled datasets dictionary
        return scaled



if __name__ == '__main__':
#     download_data_day('20180112', '/scratch/savis/aeroct/downloads/MODIS_hdf/')
    pass