'''
Created on Jun 19, 2018

@author: savis
'''
import os

if __name__ == '__main__':
    
    dl_dir = '/scratch/savis/aeroct/downloads/MODIS_hdf/'
    prd_day = 'MOD04_L2/2018/012/'
    
    filepath = '/scratch/savis/aeroct/MODIS_hdf/MOD04_L2/2018/012/MOD04_L2.A2018012.2325.061.2018013074717.hdf'
    os.makedirs(dl_dir+'test/')