import numpy as np
import pandas as pd

#read basin list with gauges list
gauge_stations = pd.read_csv('data/ma_basins_gaugelist.csv', dtype={'basin_id':str})
#only keep unique basins
gauge_stations = gauge_stations.drop_duplicates(subset='basin_id').reset_index(drop=True)

#Historical
#loop through each basin and save the temperature data
for basin_id in gauge_stations['basin_id']:
    #read temperature data for this basin for corresponding gauge
    gauge_id = gauge_stations[gauge_stations['basin_id'] == basin_id]['id'].values[0]
    temperature = pd.read_csv(f'data/swg_data/swg_output/processed/gauge_temp/1/{gauge_id}_scenario1.csv')
    #save the temperature data to a csv file
    temperature.to_csv(f'data/temperature/temp{basin_id}.csv', index=False)

#Future
#loop through each basin and save the temperature data
for basin_id in gauge_stations['basin_id']:
    #read temperature data for this basin for corresponding gauge
    gauge_id = gauge_stations[gauge_stations['basin_id'] == basin_id]['id'].values[0]
    temperature = pd.read_csv(f'data/swg_data/swg_output/processed/gauge_temp/11/{gauge_id}_scenario11.csv')
    #save the temperature data to a csv file
    temperature.to_csv(f'data/future/future_temperature/future_temp{basin_id}.csv', index=False)