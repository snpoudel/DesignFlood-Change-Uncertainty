import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#save the basin wise precipitation data to a csv file
basin_id = '01108000'
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
    
#make exceedance probability plot
def make_nep_plot(data_hist, data_future):
    sorted_data = np.sort(data_hist)
    sorted_probability = np.arange(1,len(sorted_data)+1)/len(sorted_data)
    plt.figure(figsize=(4,3))
    plt.plot(sorted_probability, sorted_data, marker='o', linestyle='-',
             color='blue', markersize=0, linewidth=1.5, label='Historical', alpha=0.8)
    sorted_data = np.sort(data_future)
    sorted_probability = np.arange(1,len(sorted_data)+1)/len(sorted_data)
    plt.plot(sorted_probability, sorted_data, marker='o', linestyle='--',
             color='red', markersize=0, linewidth=1.5, label='Future', alpha=0.8)
    plt.xlabel('Non-Exceedance Probability')
    plt.ylabel('Precipitation (mm/day)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

make_nep_plot(data_hist=stn_precip_hist['PRCP'], data_future=stn_precip_future['PRCP'])