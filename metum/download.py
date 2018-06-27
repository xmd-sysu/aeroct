'''
This module is used to retrieve forecast files contain aerosol optical depth data from
the MASS server. These are then saved locally so that the data may be processed. 

Created on Jun 21, 2018

@author: savis

TODO: Extract from MASS using Python rather than BASH script.
TODO: Allow files to be extracted from before 2015-02-03.
'''

import os
from datetime import datetime, timedelta

ext_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/global-nwp/'


def extract_from_mass(date, out_path):
    '''
    Use 'moo select' to retrieve the files from mass
    '''
    hours = [0, 6, 12, 18]                                      # forecast hours
    year = str(datetime.strptime(date, '%Y%m%d').year)          # query year
    src_uri = 'moose:/opfc/atm/global/prods/{}.pp/'.format(year)# source URI
    extract_dir = out_path                                      # extract path
    q_um = '{}aod_um.query'.format(extract_dir)                 # query file path
    
    # Make the extract directory if it does not exist and remove an existing query file.
    os.system('mkdir -p {}'.format(extract_dir))
    os.system('rm -f {}'.format(q_um))
    
    # Extract UM aod PP file from MASS
    # Update moose query file to extract AOD diagnostics at the given forecast hours:
    for h in hours:
        hh = str(h).zfill(2)
        with open(q_um, 'w') as file_writer:
            file_writer.write('begin\n')
            file_writer.write('  stash=2422\n')       # stash code for AOD
            file_writer.write('  lbft=[0..24]\n')     # forecast lead time (hours)
            file_writer.write('  lbuser_5=3\n')       # pseudo level for wavelengths
            file_writer.write('  min=[0..29]\n')      # 
#             file_writer.write('  pp_file="prods_op_gl-up_{0}_{1}*"\n'.format(date, hh) # update runs
            file_writer.write('  pp_file="prods_op_gl-mn_{0}_{1}*"\n'.format(date, hh)) # main run
            file_writer.write('end')
    
    print('Extracting UM AOD diagnostics from MASS for {}'.format(date))
    os.system('moo select {0} -f {1} {2}'.format(q_um, src_uri, extract_dir))


def download_data_day(date, out_path=None):
    '''
    Download the AOD forecast data for the given date from MASS. The location of the
    saved files can be chosen.
    
    Parameter:
    date: (str) Date to download forecast data in format "YYYYMMDD".
    out_path: (str, optional) The directory in which to save the output files.
        Default: /scratch/{USER}/aeroct/global-nwp/
    '''
    if out_path is None:
        out_path = ext_path
    
    # Check to see if the data has already been retrieved
    if len(os.popen('ls ' + out_path + '*20180610* 2> /dev/null').read()) > 1:
        print('Files already extracted.')
        return
    
    if (date == None) | (datetime.strptime(date, '%Y%m%d') > datetime(2015,02,03)):
        extract_from_mass(date, out_path)
    else:
        raise ValueError, 'Date too early to currently handle. Restrict to after 2015-02-03.'


def download_data_range(date1='20180101', date2=None, out_path=None):
    '''
    Extract all forecast files within a time frame. If date2 is not provided then all
    files since date1 are extracted.
    
    Parameters:
    date1: (str) The date to begin extracting forecast data. Format: "YYYYMMDD".
    date2: (str) The date to which files whould be extracted. Format: "YYYYMMDD".
        If None then today's date is used.
        Default: None
    out_path: (str, optional) The directory in which to save the output files.
        Default: /scratch/{USER}/aeroct/global-nwp/
        (Currently unavailable)
    '''
    if out_path is None:
        out_path = ext_path
    
    # Get the dates in a datetime type and then create a list of days
    try:
        date1_dt = datetime.strptime(date1, '%Y%m%d')
    except:
        print('date1 unable to convert to datetime. \n' +
              'Ensure the date is in the format: "YYYYMMDD"')
        return
    
    if date2 == None:
        date2_dt = datetime.utcnow()
    else:
        try:
            date2_dt = datetime.strptime(date2, '%Y%m%d')
        except:
            print('date2 unable to convert to datetime. \n' +
                  'Ensure the date is in the format: "YYYYMMDD"')
            return
    
    date_list = [date1_dt + timedelta(days=x) for x in range((date2_dt-date1_dt).days + 1)]
    
    for d in date_list:
        download_data_day(str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2))


if __name__ == '__main__':
    download_data_range(date1='20180524')