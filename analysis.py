'''
Created on Jul 3, 2018

@author: savis
'''

scratch_path = os.popen('echo $SCRATCH').read().rstrip('\n') + '/aeroct/'

def pearson_time_trend(data_set1, data_set2, dates, dir_path=scratch_path+'data_frames/'):
    dates = [datetime(2012,6,20) + timedelta(days=d) for d in np.arange(0,5,2)]
    data_set1 = 'aeronet'
    data_set2 = 'modis'
    R = np.zeros_like(dates)
    
    for i, date in enumerate(dates):
    date = date.strftime('%Y%m%d')
    df1 = aeroct.load(data_set1, date)
    df2 = aeroct.load(data_set2, date)
    match_df = aeroct.collocate(df1, df2)
    match_df.dump()
    R[i] = match_df.R
    
    plt.figure()
    plt.plot(dates, R)
    plt.show()