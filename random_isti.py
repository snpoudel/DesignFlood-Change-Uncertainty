import pandas as pd
import numpy as np
#read a list of all stations for MA 2000-2020
stn = pd.read_csv('data/MA_stations_2000-2020.csv')
#read the basin list
basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv', dtype={'basin_id':str})
# basin_id = '01108000'
#read gauging station in each basin
all_station = pd.DataFrame()
for basin_id in basin_list['basin_id']:
    stn = pd.read_csv(f'data/num_gauge_precip_basinwise_2000-2020/basin_{basin_id}.csv')
    #make a temp dataframe and keep id, lat, lon, and basin_id
    temp_df = pd.DataFrame({'id':stn['id'], 'lat':stn['lat'], 'lon':stn['lon'], 'basin_id':basin_id})
    all_station = pd.concat([all_station, temp_df], axis=0)
#only keep unique entries
all_station = all_station.drop_duplicates()
#keep lat and lon to three decimal places
all_station['lat'] = all_station['lat'].round(3)
all_station['lon'] = all_station['lon'].round(3)

#save only the id with lat and lon
new_df = all_station[['id', 'lat', 'lon']]
new_df = new_df.drop_duplicates().reset_index(drop=True)
#sort new_df by id in ascending order
new_df = new_df.sort_values('id').reset_index(drop=True)
new_df.to_csv('data/swg_data/MA_gaugelist.csv', index=False)

#save
all_station = all_station.sort_values('id').reset_index(drop=True)
all_station.to_csv('data/swg_data/MA_gaugelist_withbasins.csv', index=False)

#conver the basin id category to 1 2 3 .. category
all_station['cat_basin_id'] = all_station['basin_id'].astype('category').cat.codes
#make new dataframe, where same rows with different cat_basin_id are merged into one row with cat_basin_id as list
new_df = all_station.groupby(['id', 'lat', 'lon'])['cat_basin_id'].apply(list).reset_index()
#order by lat in ascending order
new_df = new_df.sort_values('id').reset_index(drop=True)
#save
new_df.to_csv('data/swg_data/MA_gaugelist_withbasins_cat.csv', index=False)