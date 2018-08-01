'''
Created on Jul 10, 2018

@author: savis
'''
from __future__ import division, print_function
import os
import glob
from datetime import datetime, timedelta
import numpy as np
try:
    from pyhdf.SD import SD, SDC
    from pyhdf.error import HDF4Error
    h4err = None
except ImportError as h4err:
    pass


def download_hdf_range(initial_date, days, dl_dir, satellite='Both', dl_again=False):
    '''
    Download HDF files for a range of dates beginning at initial_date. The days argument
    is a list for which each element gives the number of days after initial date for each
    day of downloaded files.
    
    Parameters:
    initial_date : str
        The date corresponding to days=0. Format is 'YYYYMMDD'.
    days : integer array
        The days after initial_date to return datetime objects.
    dl_dir : str
        The directory in which the HDF files will be saved when downloaded.
    satellite : {'Both', 'Terra, 'Aqua'}, optional (Default: 'Both')
        Which satellite's data to load.
    '''
    initial_date = datetime.strptime(initial_date, '%Y%m%d')
    dt_list = [initial_date + timedelta(days=int(d)) for d in days]
    
    for date in dt_list:
        print('Downloading for: ', date)
        download_hdf_day(date, dl_dir, satellite)


def download_hdf_day(date, dl_dir, satellite='Both', dl_again=False):
    src_url='https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/'
    
    print('Downloading MODIS ({0}) HDF files for {1}.'.format(satellite, date))
    
    # Make dl_dir if the directory does not exist
    if not os.path.exists(dl_dir):
        os.makedirs(dl_dir)
    
    # Get an iterable of the codes for the satellites used in the urls 
    if satellite == 'Both':     sat_codes = ['MOD04_L2', 'MYD04_L2']
    elif satellite == 'Terra':  sat_codes = ['MOD04_L2']
    elif satellite == 'Aqua':   sat_codes = ['MYD04_L2']
    
    # Convert date format from yyyymmdd to yyyy/jjj for use in the url
    if type(date) is str:
        date = datetime.strptime(date, '%Y%m%d')
    date_yj = date.strftime('%Y/%j')
    
    for sat in sat_codes:
        dir_url = src_url + sat + '/' + date_yj
        
        ### DOWNLOAD USING curl_dir
        query = 'curl_dir -fx .hdf {0} {1}'.format(dir_url, dl_dir)
        print(query)
        os.system(query)
        
        #### DOWNLOAD USING URLLIB2 ONE AT A TIME
#         # Get list of filenames for the date
#         filename_pattern = sat + '\.' + '.+?.hdf'
#         try:
#             dir_r = urlopen(Request(dir_url))
#         except URLError, e:
#             print(e.reason)
#         filenames = re.findall(filename_pattern, dir_r.read())
#         
#         # Download hdf files
#         for filename in filenames:
#             print('.', end='')
#             if (not os.path.exists(dl_dir + filename)) | dl_again:
#                 num_downloaded += 1
#                 file_url = dir_url + filename
#                 
#                 try:
#                     with closing(urlopen(file_url)) as file_r:
#                         with open(dl_dir + filename, 'wb') as write:
#                             copyfileobj(file_r, write)
#                 except URLError, e:
#                     print(e.reason)
#         print()
#         
#     if num_downloaded > 0:
#         print('Download complete - {0} files downloaded to: {1}'.format(str(num_downloaded),
#                                                                       dl_dir))
#     else:
#         print('Files already exist in {0}'.format(dl_dir))


def load_data_day(date, dl_dir, satellite='Both', dl_again=False, keep_files=True):
    '''
    This function can be used to download MODIS data for a day. A dictionary is returned
    containing 1D arrays with the following fields:
    'LNGT': longitudes,
    'LTTD': latitudes,
    'AOD_NM550' : AOD,
    'ARSL_TYPE': aerosol type (dust=1),
    'ARSL_RTVL_CNFC_FLAG': quality flag (0
    'YEAR', 'MNTH', 'DAY', 'HOUR', 'MINT': times,
    'AE_LAND', 'SSA_LAND', 'FM_FRC', 'FM_FRC_OCEAN', 'AE_OCEAN'
    
    Parameters:
    date : str 
        The date for which to retrieve records. Format: YYYYMMDD.
    dl_dir : str
        The directory in which the HDF files will be saved when downloaded.
    satellite : {'Both', 'Terra, 'Aqua'}, optional (Default: 'Both')
        Which satellite's data to load.
    dl_again : bool, optional (Default: False)
        If it is True then it will download the data again, even if the file already
        exists.
    keep_files : bool, optional (Default: True)
        If False then the HDF files will be deleted once they have been passed into
        Python. If there are no more files in dl_dir once the files have been removed
        then the directory will also be removed.
    '''
    
    # Convert date format from yyyymmdd to yyyyjjj to find the HDF filepaths
    date_dt = datetime.strptime(date, '%Y%m%d')
    date_yj = date_dt.strftime('%Y%j')
    
    # Get the satellite code for the desired satellite data for use in the filenames
    if satellite == 'Both':     sat_code = ''
    elif satellite == 'Terra':  sat_code = 'MOD04_L2'
    elif satellite == 'Aqua':   sat_code = 'MYD04_L2'    
    
    files = glob.glob('{0}*{1}*{2}.*.hdf'.format(dl_dir, sat_code, date_yj))
    
    # Download if files not downloaded (<100 files) or dl_again is True
    if dl_again | (len(files) <= 100):
        download_hdf_day(date, dl_dir, satellite, dl_again)
        files = glob.glob('{0}*{1}*{2}.*.hdf'.format(dl_dir, sat_code, date_yj))
    
    # Get the fields from the files and concatenate them in lists
    lon, lat, time, asl_type, aod, sat_idny = [], [], [], [], [], []
    ae_land, ssa_land, fmf_land, fmf_ocean, ae_ocean = [], [], [], [], []
    fieldnames = ['Longitude', 'Latitude', 'Scan_Start_Time', 'Aerosol_Type_Land', 
                  'AOD_550_Dark_Target_Deep_Blue_Combined',
                  'AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag',
                  'Deep_Blue_Angstrom_Exponent_Land',
                  'Deep_Blue_Spectral_Single_Scattering_Albedo_Land',
                  'Optical_Depth_Ratio_Small_Land',
                  'Optical_Depth_Ratio_Small_Ocean_0.55micron',
                  'Angstrom_Exponent_1_Ocean']
    
    print('Loading {0} MODIS HDF files for {1}.'.format(len(files), date))
    
    for f in files[::-1]:
        print('.', end='')
        try:
            parser = h4Parse(f)
            scaled = parser.get_scaled(fieldnames)
        except HDF4Error:
            print('Issue loading file, skipping this file: {0}'.format(f))
        
        # Convert 'Scan_Start_Time' (seconds since 1993-01-01)
        # to hours since 00:00:00 on date
        date_hours = (date_dt - datetime(1993,1,1)).days * 24
        time_hours = scaled['Scan_Start_Time'] / 3600 - date_hours
        
        # Get the satellite ID number
        if 'MOD04_L2' in f:     sat_id = 783
        elif 'MYD04_L2' in f:   sat_id = 784
        # Include only the data with the highest quality flag
        highest_qf = (scaled['AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag'] == 3)
        
        lon.extend(scaled['Longitude'][highest_qf])
        lat.extend(scaled['Latitude'][highest_qf])
        time.extend(time_hours[highest_qf])
        aod.extend(scaled['AOD_550_Dark_Target_Deep_Blue_Combined'][highest_qf])
        sat_idny.extend(np.full_like(time_hours[highest_qf], sat_id))
        asl_type.extend(scaled['Aerosol_Type_Land'][highest_qf])
        ae_land.extend(scaled['Deep_Blue_Angstrom_Exponent_Land'][highest_qf])
        ssa_land.extend(scaled['Deep_Blue_Spectral_Single_Scattering_Albedo_Land'][1, highest_qf])
        fmf_land.extend(scaled['Optical_Depth_Ratio_Small_Land'])
        fmf_ocean.extend(scaled['Optical_Depth_Ratio_Small_Ocean_0.55micron'][highest_qf])
        ae_ocean.extend(scaled['Angstrom_Exponent_1_Ocean'][highest_qf])
    
    lon = np.array(lon)
    lat = np.array(lat)
    time = np.array(time)
    aod = np.array(aod)
    sat_idny = np.array(sat_idny, dtype=int)
    asl_type = np.array(asl_type)
    ae_land = np.array(ae_land)
    ssa_land = np.array(ssa_land)
    fmf_land = np.array(fmf_land)
    fmf_ocean = np.array(fmf_ocean)
    ae_ocean = np.array(ae_ocean)
    print()
    
    # Get fine mode fraction for both land and ocean
    fmf = fmf_land
    print(fmf[(fmf<0)|(fmf>1)])
    fmf[fmf_land < 0] = fmf_ocean[fmf_land < 0]
     
    # Put all of the fields into one structured array
    fields_type = np.dtype([('AOD_NM550', aod.dtype), ('LNGD', lon.dtype),
                            ('LTTD', lat.dtype), ('TIME', time.dtype),
                            ('STLT_IDNY', sat_idny.dtype), ('ARSL_TYPE', asl_type.dtype),
                            ('AE_LAND', ae_land.dtype), ('SSA_LAND', ssa_land.dtype),
                            ('FM_FRC_LAND', fmf_land.dtype),
                            ('FM_FRC_OCEAN', fmf_ocean.dtype), ('AE_OCEAN', ae_ocean.dtype)])
    
    fields_arr = np.empty(len(lon), dtype = fields_type)
    fields_arr['AOD_NM550'] = aod
    fields_arr['LNGD'] = lon
    fields_arr['LTTD'] = lat
    fields_arr['TIME'] = time
    fields_arr['STLT_IDNY'] = sat_idny
    fields_arr['ARSL_TYPE'] = asl_type
    fields_arr['AE_LAND'] = ae_land
    fields_arr['SSA_LAND'] = ssa_land
    fields_arr['ARSL_SMAL_MODE_FRCN'] = fmf
    fields_arr['FM_FRC_OCEAN'] = fmf_ocean
    fields_arr['AE_OCEAN'] = ae_ocean
    
    # Remove files?
    if keep_files == False:
        for f in files:
            os.remove(f)
        if not os.listdir(dl_dir):
            os.rmdir(dl_dir)
    
    return fields_arr


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
    download_hdf_range('20170801', np.arange(242, dtype=int), '/scratch/savis/aeroct/downloads/MODIS/hdf', satellite='Both')
#     download_hdf_day('20180604', '/scratch/savis/aeroct/downloads/MODIS/hdf', dl_again=True, satellite='Aqua')