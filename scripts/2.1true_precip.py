import numpy as np
import pandas as pd
import geopandas as gpd
from mpi4py import MPI

#set up MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#rad filtered basin with at least 2 gauges
basin_list = pd.read_csv('data/ma29basins.csv', dtype={'basin_id':str})
basin_id = basin_list['basin_id'][rank] #run 31 processess
#basid id
#basin_id = '01108000'
#read basin shapefile

basin_shapefile = gpd.read_file(f'data/prms_drainage_area_shapes/model_{basin_id}_nhru.shp')
#convert the basin shapefile to the same coordinate system as the stations
basin_shapefile = basin_shapefile.to_crs(epsg=4326)

##Divide the basin into meshgrid of size 0.125 degree
#find the bounding box of the basin
grid_size = 0.0625 #0.0625
minx, miny, maxx, maxy = basin_shapefile.total_bounds
#create meshgrid
x = np.arange(minx, maxx, grid_size)
y = np.arange(miny, maxy, grid_size)
#meshgrid
xx, yy = np.meshgrid(x, y)
#flatten the meshgrid
xx = xx.flatten()
yy = yy.flatten()
#create a dataframe
df_grid = pd.DataFrame({'lon':xx,'lat':yy})
#only keep the points that lies within the basin
gdf_grid = gpd.GeoDataFrame(df_grid,geometry=gpd.points_from_xy(df_grid['lon'],df_grid['lat']))
gdf_grid = gdf_grid[gdf_grid.within(basin_shapefile.unary_union)].reset_index(drop=True)

#read all gauging stations for this basin
gauge_stations = pd.read_csv('data/ma_basins_gaugelist.csv', dtype={'basin_id':str})
gauge_stations = gauge_stations[gauge_stations['basin_id'] == basin_id]

#read all gauge stations data for this basin and merge them
precip_all_stations = pd.DataFrame()
for gauge_id in gauge_stations['id']:
    gauge_data = pd.read_csv(f'data/swg_data/swg_output/processed/gauge_precip/1/{gauge_id}_scenario1.csv')
    #add the station id to the dataframe
    gauge_data['STATION'] = gauge_id
    #append the data to the dataframe
    precip_all_stations = pd.concat([precip_all_stations, gauge_data], ignore_index=True)

#add the lat and lon of the stations to precip_all_stations from gauge_stations
precip_all_stations = pd.merge(precip_all_stations, gauge_stations[['id','lat','lon']], left_on='STATION', right_on='id', how='left') 

#loop through each day, calculate inverse distance weighted precipitation for each grid point in the basin
precip_df = pd.DataFrame() 
unique_dates = precip_all_stations['date'].unique()
# unique_dates = np.array([np.datetime64(date) for date in unique_dates])
# unique_dates = np.sort(unique_dates)
for date in unique_dates:
    #get the precipitation data for that day
    date = str(date)
    precip_day = precip_all_stations[precip_all_stations['date'] == date].reset_index(drop=True)
    precip_grid = [] #store the precipitation data for each grid point
    #loop through each grid point in the basin
    for i in range(len(gdf_grid)):
        #get the lat and lon of the grid point
        lat = gdf_grid['lat'][i]
        lon = gdf_grid['lon'][i]
        #calculate the distance between the grid point and all stations
        precip_day['distance'] = np.sqrt((precip_day['lat'] - lat)**2 + (precip_day['lon'] - lon)**2)
        #calculate the inverse distance weighted precipitation
        precip_day['weight'] = 1/(precip_day['distance']**2)
        precip_day['precip_weighted'] = precip_day['prcp']*precip_day['weight']
        precip_weighted = precip_day['precip_weighted'].sum()/precip_day['weight'].sum()
        #append the precipitation data of each grid
        precip_grid.append(precip_weighted)
    #find mean precipitation for a basin across all grid points for that day
    precip_mean = np.mean(precip_grid)
    #round to 4 digits
    precip_mean = round(precip_mean, 3)
    precip_temp_df = pd.DataFrame({'DATE':date, 'PRECIP':[precip_mean]})
    precip_df = pd.concat([precip_df, precip_temp_df], ignore_index=True)

#save the basin wise precipitation data to a csv file
precip_df.to_csv(f'data/true_precip/true_precip{basin_id}.csv', index=False)
#also save the result in interpolated precip folder with 99 as coverage and 5 combinations(0,1,2,3,4)
#these precipitation sets will be forced to models to see how model output can vary even under same input
precip_df.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage99_comb0.csv', index=False)
precip_df.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage99_comb1.csv', index=False)
precip_df.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage99_comb2.csv', index=False)
precip_df.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage99_comb3.csv', index=False)
precip_df.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage99_comb4.csv', index=False)
precip_df.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage99_comb5.csv', index=False)
precip_df.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage99_comb6.csv', index=False)
precip_df.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage99_comb7.csv', index=False)
precip_df.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage99_comb8.csv', index=False)
precip_df.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage99_comb9.csv', index=False)