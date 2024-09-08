import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv', sep='\t', dtype={'basin_id':str})

#basin_id = '01108000'
for basin_id in basin_list['basin_id']:
    #read all stations for this basin
    gauge_stns = pd.read_csv(f'data/num_gauge_precip_basinwise_2000-2020/basin_{basin_id}.csv')

    #Generate synthetic precipitation under climate change for each gauging station in the basin
    for stn in gauge_stns['id']:
        #read processed precipitation time series for this gauuging station
        stn_precip_hist = pd.read_csv(f'data/preprocessed_gauge_precip_2000-2020/{stn}.csv')
        #make synthetic precipitation under climate change by
        #  increasing the top 25% of the precipitation by 10%, and reducing the bottom 25% by 10%
        stn_precip_future = stn_precip_hist.copy()
        stn_precip_future['PRCP'] = np.where(stn_precip_hist['PRCP'] > stn_precip_hist['PRCP'].quantile(0.99),
                                        stn_precip_hist['PRCP']*1.5, stn_precip_hist['PRCP']*0.8) 
        #save the synthetic precipitation to a csv file
        stn_precip_future.to_csv(f'data/future/future_gauge_precip/{stn}.csv', index=False)
        