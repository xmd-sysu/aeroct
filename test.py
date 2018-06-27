'''
Created on Jun 19, 2018

@author: savis
'''

import os, sys
from urllib2 import urlopen, Request, URLError
import ftplib
import data_frame
#import metum.download, metum.process

from metdb import obs
import pwd
from datetime import datetime 

# sys.path.insert(3, '/net/home/h05/fra6/myPython/ypy/')
# from ypylib import mdbx

if __name__ == '__main__':
    
#     metum.download.download_data_range(date1='20180620')
#     cube = metum.process.load_files(24)
#     df = metum.process.process_data(cube, 24)
#     print(df)
    
#     ext_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/global-nwp/'
#     print(len(os.popen('ls ' + ext_path + '*20180610* 2> /dev/null').read()))
    
    # url    =    ftp:// ladsftp.nascom.nasa.gov/allData/51 / MYD04_L2 / YYYY/jjj
#     try:
#         r = urlopen(Request(url))
#     except URLError, e:
#         print(e.reason)
#     print(r.readlines())
#     ftp = ftplib.FTP('ladsftp.nascom.nasa.gov')
#     ftp.cwd('/allData/51/MYD04_L2/2016/001')
#     data = []
#     ftp.dir(data.append)
#     print(data)
    
#     elements = ['YEAR', 'MNTH', 'DAY', 'HOUR', 'MINT', 'SCND', 'AOD_NM550']
#     query = mdbx.Query('SATAOD', elements)
#     query.plot('AOD_NM550', show=True)
    
    start = datetime(2018, 06, 24, 06, 00).strftime('%Y%m%d/%H%MZ')
    stop = datetime(2018, 06, 24, 12, 00).strftime('%Y%m%d/%H%MZ')
    user = os.getenv('USER')
    contact = '{}@metoffice.gov.uk'.format(pwd.getpwnam(user).pw_gecos)
    keywords = ['START TIME ' + start, 'END TIME ' + stop]
    elements = ['YEAR', 'MNTH', 'DAY', 'HOUR', 'MINT', 'LTTD', 'LNGD', 'AOD_NM550']
    data = obs(contact, 'SATAOD', keywords, elements, hostname=None)
    print([datetime(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]) for i in range(len(data))])