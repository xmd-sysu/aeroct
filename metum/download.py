'''
This module is used to retrieve forecast files contain aerosol optical depth data from
the MASS server. These are then saved locally so that the data may be processed. 

Created on Jun 21, 2018

@author: savis

TODO: Allow files to be extracted from before 2015-02-03.
'''

import os
import glob
from datetime import datetime, timedelta
import numpy as np

ext_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/downloads/UM/pp/'


def extract_from_mass(dates, fc_time, extract_dir):
    '''
    Use 'moo select' to retrieve the files from MASS for a list of dates in one year.
    Dates must be a list of strings in format 'YYYYMMDD'.
    '''
    hours = [0, 6, 12, 18]                                        # forecast hours
    year = str(datetime.strptime(dates[0], '%Y%m%d').year)        # query year
    src_uri = 'moose:/opfc/atm/global/prods/{0}.pp/'.format(year) # source URI
    q_um = '{0}aod_um.query'.format(extract_dir)                  # query file path
    
    # Make the extract directory if it does not exist and create a new query file.
    os.system('mkdir -p {0}'.format(extract_dir))
    os.system('rm -f {0}'.format(q_um))
    os.system('touch {0}'.format(q_um))
    
    # Extract UM aod PP file from MASS
    # Update moose query file to extract AOD diagnostics at the given forecast hours:
    
    with open(q_um, 'w') as file_writer:
        for d in dates:
            for h in hours:
                hh = str(h).zfill(2)
                file_writer.write('begin\n')
                file_writer.write('  stash=2422\n')       # stash code for AOD
                file_writer.write('  lbft={0}\n'.format(fc_time))     # forecast lead time (hours)
                file_writer.write('  lbuser_5=3\n')       # pseudo level for wavelengths
                file_writer.write('  min=[0..29]\n')      # 
    #             file_writer.write('  pp_file="prods_op_gl-up_{0}_{1}*"\n'.format(d, hh)) # update runs
                file_writer.write('  pp_file="prods_op_gl-mn_{0}_{1}*"\n'.format(d, hh)) # main run
                file_writer.write('end\n\n')
    
    print('Extracting {0} day(s) of UM AOD files from {1} to location {2}'\
          .format(len(dates), dates[0], extract_dir))
    os.system('moo select {0} -f {1} {2}'.format(q_um, src_uri, extract_dir))


def download_data_day(date, forecast_time, out_path=None, dl_again=False):
    '''
    Download the AOD forecast data for the given date from MASS. The location of the
    saved files can be chosen.
    
    Parameter:
    date : str
        Date to download forecast data in format "YYYYMMDD".
    forecast_time : int
        The forecast lead time.
        Possible choices: 0, 3, 6, 9, 12, 15, 18, 21, 24.
    out_path : str, optional (Default: /scratch/{USER}/aeroct/downloads/UM/pp/)
        The directory in which to save the output files.
    dl_again : bool, optional (Default: False)
        Choose whether to extract the files if they any are detected in the output
        directory.
    '''
    if datetime.strptime(date, '%Y%m%d') < datetime(2015,02,03):
        raise ValueError('Date too early to currently handle. Restrict to after 2015-02-03.')
    
    if out_path is None:
        out_path = ext_path
    fc = str(forecast_time).zfill(3)
    
    # Get the dates of the two files containing data during 'date'
    days_before = int((forecast_time - 6) / 24)
    date1 = datetime.strptime(date, '%Y%m%d') - timedelta(days=(days_before + 1))
    date2 = datetime.strptime(date, '%Y%m%d') - timedelta(days=days_before)
    date1 = date1.strftime('%Y%m%d')
    date2 = date2.strftime('%Y%m%d')
    
    # Download the data for each of the dates
    day1_query = 'ls {0}*{1}*_{2:03d}* 2> /dev/null'.format(out_path, date1, forecast_time+3)
    day2_query = 'ls {0}*{1}*_{2:03d}* 2> /dev/null'.format(out_path, date2, forecast_time+3)
    downloaded_data = False
    
    if (len(os.popen(day1_query).read()) <= 222) | (dl_again == True):
        extract_from_mass([date1], fc, out_path)
        downloaded_data = True
    
    if (len(os.popen(day2_query).read()) <= 222) | (dl_again == True):
        extract_from_mass([date2], fc, out_path)
        downloaded_data = True
        
    if downloaded_data:
        print('UM AOD file extraction complete.\n')
    else:
        print('UM AOD files already extracted.')


def download_data_range(dates, forecast_time=0, dl_dir=None, dl_again=False):
    '''
    Extract all forecast files within a time frame. If date2 is not provided then all
    files since date1 are extracted.
    
    Parameters:
    dates : str list or datetime list
        The dates to extract forecast data. Format: "YYYYMMDD" for strings.
    forecast_time : int, optional (Default: 0)
        The forecast lead time. Possible choices: 0, 3, 6, 9, 12, 15, 18, 21, 24.
    dl_dir : str, optional (Default: /scratch/{USER}/aeroct/downloads/UM/pp/)
        The directory in which to save the output files.
    '''
    if dl_dir is None:
        dl_dir = ext_path
    
    if isinstance(dates[0], str):
        dates = [datetime.strptime(date, '%Y%m%d') for date in dates]
    
    # Get the dates of the files containing the data for each given date
    dates.sort()
    for i_d, date in enumerate(dates):
        days_before = int((forecast_time - 6) / 24)
        dates[i_d] = date - timedelta(days=(days_before + 1))
    dates.append(dates[-1] + timedelta(days=1))
    
    # Find the dates for which data exists and remove these dates from the list
    if not dl_again:
        new_dates = []
        for date in dates:
            files = glob.glob('{0}*{1}*_{2}.*'.format(dl_dir, date,
                                                      str(forecast_time+3).zfill(3)))
            if len(files) == 0:
                new_dates.append(date)
        dates = new_dates
    
    # Split the dates into years and then extract from MASS
    years = np.array([date.year for date in dates])
    for y in np.unique(years):
        dates_y = [dates[i] for i in range(len(dates)) if years[i]==y]
        dates_y_str = [date.strftime('%Y%m%d') for date in dates_y]
        extract_from_mass(dates_y_str, forecast_time, dl_dir)


if __name__ == '__main__':
    days = np.arange(41)
    initial_date = datetime(year=2018, month=1, day=1)
    dates = [initial_date + timedelta(days=int(d)) for d in days]
    
    download_data_range(dates, forecast_time=24)
    download_data_range(dates, forecast_time=30)
    download_data_range(dates, forecast_time=36)
    download_data_range(dates, forecast_time=42)