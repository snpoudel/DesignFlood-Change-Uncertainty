import numpy as np
import pandas as pd
import geopandas as gpd
from mpi4py import MPI

#set up MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#rad filtered basin with at least 2 gauges
basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv', sep='\t', dtype={'basin_id':str})
basin_id = basin_list['basin_id'][rank] #run 32 processess
#basid id
#basin_id = '01108000'
#read basin shapefile

basin_shapefile = gpd.read_file(f'data/prms_drainage_area_shapes/model_{basin_id}_nhru.shp')
#convert the basin shapefile to the same coordinate system as the stations
basin_shapefile = basin_shapefile.to_crs(epsg=4326)

##Divide the basin into meshgrid of size 0.125 degree
#find the bounding box of the basin
grid_size = 0.03125 #0.03125
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

#read basins with number of gauge stations file
stn = pd.read_csv(f'data/num_gauge_precip_basinwise_2000-2020/basin_{basin_id}.csv')

#Read all precip data
precip_all_stations = pd.read_csv('data/preprocessed_all_stations_precip_2000-2020.csv')
#only keep stations that are within the basin
precip_keep = precip_all_stations[precip_all_stations['STATION'].isin(stn['id'])]

#loop through each day, calculate inverse distance weighted precipitation for each grid point in the basin
precip_df = pd.DataFrame() 
for date in precip_keep['DATE'].unique():
    #get the precipitation data for that day
    precip_day = precip_keep[precip_keep['DATE'] == date].reset_index(drop=True)
    precip_grid = [] #store the precipitation data for each grid point
    #loop through each grid point in the basin
    for i in range(len(gdf_grid)):
        #get the latitude and longitude of the grid point
        lat = gdf_grid['lat'][i]
        lon = gdf_grid['lon'][i]
        #calculate the distance between the grid point and all stations
        precip_day['distance'] = np.sqrt((precip_day['LATITUDE'] - lat)**2 + (precip_day['LONGITUDE'] - lon)**2)
        #calculate the inverse distance weighted precipitation
        precip_day['weight'] = 1/(precip_day['distance']**2)
        precip_day['precip_weighted'] = precip_day['PRCP']*precip_day['weight']
        precip_weighted = precip_day['precip_weighted'].sum()/precip_day['weight'].sum()
        #append the precipitation data of each grid
        precip_grid.append(precip_weighted)
    #find mean precipitation for a basin across all grid points for that day
    precip_mean = np.mean(precip_grid)
    #round to 4 digits
    precip_mean = round(precip_mean, 4)
    precip_temp_df = pd.DataFrame({'DATE':date, 'PRECIP':[precip_mean]})
    precip_df = pd.concat([precip_df, precip_temp_df], ignore_index=True)

#save the basin wise precipitation data to a csv file
precip_df.to_csv(f'data/true_precip/true_precip{basin_id}.csv', index=False)