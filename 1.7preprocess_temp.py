#read the mean areal temperature data for 29 MA basins and assign this data to all the stations in the basin
import pandas as pd #to work with pandas dataframe

#read gaugelist and basinlist csv files
file = pd.read_csv(f'data/swg_data/MA_gaugelist_withbasins.csv', dtype={'basin_id':str})
unique_gauges = file['id'].unique()
for id in unique_gauges:
    #get the basin_id for the station
    basin_id = file[file['id'] == id]['basin_id'].values[0]
    #read the mean areal temperature data for the basin
    basin_temp = pd.read_csv(f'data/processed-era5-temp/temp_{basin_id}.csv')
    basin_temp['time'] = pd.to_datetime(basin_temp['time'])
    basin_temp = basin_temp[(basin_temp['time'] >= '2000-01-01') & (basin_temp['time'] <= '2020-12-31')]
    #save the temperature data for the gauging station
    basin_temp.to_csv(f'data/gauge_temp/temp_{id}.csv', index=False)
