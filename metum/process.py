'''
This module loads the files retrieved using the download module and processes the data
so that it may easily be passed into a data_frame class. This will allow it to be more
easily be compared with other data sources.

Created on Jun 22, 2018

@author: savis
'''
from __future__ import division
import os
import iris
import numpy as np
from datetime import datetime, timedelta
from iris.experimental.equalise_cubes import equalise_attributes

scratch_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/downloads/UM/pp/'
wavelength = {3 : 550}  # Wavelengths given the pseudo-level


def load_files(date, forecast_time, src_path=None):
    '''
    Load the files with the chosen forecast time as an iris cube and select the desired
    wavelength. The files are merged along time when they are loaded.
    
    Parameters:
    date: (str) The date of the data to load in format 'YYYYMMDD'.
    forecast_time: (int) The number of hours ahead for the forecast.
        Possible choices: 0, 3, 6, 9, 12, 15, 18, 21, 24.
    src_path: (Optional) (str) The file path containing the extracted forecast files.
        Default: '/scratch/{USER}/aeroct/global-nwp/'
    '''
    if src_path is None:
        src_path=scratch_path
    
    if np.isin(np.arange(0,166,3), forecast_time).any():
        forecast_time_str = str(forecast_time+3).zfill(3)
    else:
        raise ValueError('Invalid forecast_time. It must be a multiple of 3 between 0 \
                          and 165.')
    
    # Get the dates of the two files containing data during 'date'
    days_before = int((forecast_time - 6) / 24)
    date1 = datetime.strptime(date, '%Y%m%d') - timedelta(days=(days_before + 1))
    date2 = datetime.strptime(date, '%Y%m%d') - timedelta(days=days_before)
    date1 = date1.strftime('%Y%m%d')
    date2 = date2.strftime('%Y%m%d')
    
    # This loads files from src_path with filenames containing '*YYYYMMDD*_###.*' where
    # ### is the forecast time plus 3, ie. 003 is the analysis time.
    aod_cube1 = iris.load_raw('{0}*{1}*_{2}.*'.format(src_path, date1, forecast_time_str))
    aod_cube2 = iris.load_raw('{0}*{1}*_{2}.*'.format(src_path, date2, forecast_time_str))
    
    # Equalise attributes to solve issues with merging (eg. mismatching UM version)
    equalise_attributes(aod_cube1)
    equalise_attributes(aod_cube2)
    aod_cube1 = aod_cube1.merge_cube()
    aod_cube2 = aod_cube2.merge_cube()
    
    # Concatenate the two days into a single cube
    cube_list = iris.cube.CubeList([aod_cube1, aod_cube2])
    for cube in cube_list:
        cube.remove_coord('forecast_period')
        forecast_coord = iris.coords.DimCoord(forecast_time, units='hours', 
                                              standard_name='forecast_period')
        cube.add_aux_coord(forecast_coord)
    equalise_attributes(cube_list)
    aod_cube = cube_list.concatenate_cube()
    return aod_cube


def process_data(aod_cube, date, forecast_time):
    '''
    Process the AOD data from an iris cube into a list that may be passed into a
    data_frame so that it may be compared with other data sources.
    
    Parameter:
    aod_cube: (Iris cube) The cube loaded from the forecast files using load_files.
    date: (str, datetime) The date of the loaded files. If a string, in format 'YYYYMMDD'.
        If a datetime, ensure there is no time.
    forecast_time: (int) The number of hours ahead for the forecast.
        Possible choices: 0, 3, 6, 9, 12, 15, 18, 21, 24.
    '''
    
    if type(date) is not datetime:
        date = datetime.strptime(date, '%Y%m%d')
    
    # Change the time to be hours since 00:00:00 on date
    hours1970 = aod_cube.coord('time').points   # hours since 1970-1-1
    
    time = np.zeros_like(hours1970)
    for i_h, hour in enumerate(hours1970):
        dt = datetime(1970, 1, 1) + timedelta(hours=hour)
        time[i_h] = (dt - date).days * 24 + (dt - date).seconds / 3600
    aod_cube.coord('time').points = time
    
    # Add meta-data to cube
    aod_cube.coord('forecast_period').rename('forecast_time')
    
    date_coord = iris.coords.AuxCoord(date, long_name='date')
    aod_cube.add_aux_coord(date_coord)
    
    wl = wavelength[aod_cube.coord('pseudo_level').points[0]]
    wl_coord = iris.coords.AuxCoord(wl, long_name='wavelength', units='nm')
    aod_cube.add_aux_coord(wl_coord)
     
    return aod_cube


if __name__ == '__main__':
    cube = load_files('20180602', 0)
    process_data(cube, '20180602', 0)
    print(cube)