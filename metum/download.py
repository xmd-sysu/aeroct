'''
This module is used to retrieve forecast files contain aerosol optical depth data from
the MASS server. These are then saved locally so that the data may be processed. 

Created on Jun 21, 2018

@author: savis

TODO: Allow files to be extracted from before 2015-02-03.
'''

import os
from datetime import datetime, timedelta

ext_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/downloads/UM/'


def extract_from_mass(date, fc_time, out_path):
    '''
    Use 'moo select' to retrieve the files from mass
    '''
    hours = [0, 6, 12, 18]                                      # forecast hours
    year = str(datetime.strptime(date, '%Y%m%d').year)          # query year
    src_uri = 'moose:/opfc/atm/global/prods/{0}.pp/'.format(year)# source URI
    extract_dir = out_path                                      # extract path
    q_um = '{0}aod_um.query'.format(extract_dir)                 # query file path
    
    # Make the extract directory if it does not exist and create a new query file.
    os.system('mkdir -p {0}'.format(extract_dir))
    os.system('rm -f {0}'.format(q_um))
    os.system('touch {0}'.format(q_um))
    
    # Extract UM aod PP file from MASS
    # Update moose query file to extract AOD diagnostics at the given forecast hours:
    
    with open(q_um, 'w') as file_writer:
        for h in hours:
            hh = str(h).zfill(2)
            file_writer.write('begin\n')
            file_writer.write('  stash=2422\n')       # stash code for AOD
            file_writer.write('  lbft={0}\n'.format(fc_time))     # forecast lead time (hours)
            file_writer.write('  lbuser_5=3\n')       # pseudo level for wavelengths
            file_writer.write('  min=[0..29]\n')      # 
#             file_writer.write('  pp_file="prods_op_gl-up_{0}_{1}*"\n'.format(d, hh)) # update runs
            file_writer.write('  pp_file="prods_op_gl-mn_{0}_{1}*"\n'.format(date, hh)) # main run
            file_writer.write('end\n\n')
    
    print('Extracting UM AOD files from MASS for {0} to location {1}'.format(date, extract_dir))
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
    out_path : str, optional (Default: /scratch/{USER}/aeroct/downloads/UM/)
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
    
    # Download the data for date and the day before if it has not been extracted.
    date0 = (datetime.strptime(date, '%Y%m%d') - timedelta(1)).strftime('%Y%m%d')
    day0_query = 'ls {0}*{1}*_{2:03d}* 2> /dev/null'.format(out_path, date0, forecast_time+3)
    day1_query = 'ls {0}*{1}*_{2:03d}* 2> /dev/null'.format(out_path, date, forecast_time+3)
    downloaded_data = False
    
    if (len(os.popen(day0_query).read()) <= 222) | (dl_again == True):
        extract_from_mass(date0, fc, out_path)
        downloaded_data = True
    
    if (len(os.popen(day1_query).read()) <= 222) | (dl_again == True):
        extract_from_mass(date, fc, out_path)
        downloaded_data = True
        
    if downloaded_data:
        print('UM AOD file extraction complete.\n')
    else:
        print('UM AOD files already extracted.')


def download_data_range(date1, date2=None, forecast_time=0, out_path=None, dl_again=False):
    '''
    Extract all forecast files within a time frame. If date2 is not provided then all
    files since date1 are extracted.
    
    Parameters:
    date1 : str
        The date to begin extracting forecast data. Format: "YYYYMMDD".
    date2 : str, optional (Default: None)
        The date to which files whould be extracted. Format: "YYYYMMDD".
        If None then today's date is used.
    forecast_time : int, optional (Default: 0)
        The forecast lead time. Possible choices: 0, 3, 6, 9, 12, 15, 18, 21, 24.
    out_path : str, optional (Default: /scratch/{USER}/aeroct/downloads/UM/)
        The directory in which to save the output files.
    dl_again : bool, optional (Default: False)
        Choose whether to extract the files if they any are detected in the output
        directory.
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
    print(date_list)
    
    for d in date_list:
        download_data_day(str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2),
                          forecast_time, out_path, dl_again)


if __name__ == '__main__':
    download_data_day('20180606', forecast_time=0)