#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import random
from mpi4py import MPI

#Set up communicator to parallelize job in cluster using MPI
comm = MPI.COMM_WORLD #Get the default communicator object
rank = comm.Get_rank() #Get the rank of the current process
size = comm.Get_size() #Get the total number of processes

#read basin lists with at least 2 stations
basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv',dtype={'basin_id':str})

basin_list['total_num_stn_used'] = basin_list['num_stations']-1 #use total of n-1 station for a basin
basin_list['stn_array'] = basin_list['total_num_stn_used'].apply(lambda x:np.arange(1,x+1))
basin_list = basin_list.explode('stn_array')
basin_list = basin_list.reset_index(drop=True)

#extract necessary input for each MPI jobs
basin_id = basin_list['basin_id'][rank]
total_comb = min(10, basin_list['num_stations'][rank])
num_station = basin_list['stn_array'][rank]

#basid id
#basin_id = '01108000'
#read basin shapefile
basin_shapefile = gpd.read_file(f'data/prms_drainage_area_shapes/model_{basin_id}_nhru.shp')
#convert the basin shapefile to the same coordinate system as the stations
basin_shapefile = basin_shapefile.to_crs(epsg=4326)

##Divide the basin into meshgrid of size 0.125 degree
#find the bounding box of the basin
grid_size = 0.0625 #0.03125
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
#combine all historical precipitation for this basin
precip_all_historical = pd.DataFrame()
for i in stn['id']:
    temp_df = pd.read_csv(f'data/preprocessed_gauge_precip_2000-2020/{i}.csv')
    precip_all_historical = pd.concat([precip_all_historical, temp_df], ignore_index=True)
#combine all future precipitation for this basin
precip_all_future = pd.DataFrame()
for i in stn['id']:
    temp_df = pd.read_csv(f'data/future/future_gauge_precip/{i}.csv')
    precip_all_future = pd.concat([precip_all_future, temp_df], ignore_index=True)

# Function to generate random combinations of grid cells
# This avoids use of combinations from itertools, as it is slow for large number of grid cells
def generate_random_combinations(total_stations, used_stations, num_combinations): 
    random_combinations = []
    while len(random_combinations) < num_combinations:
        combination = random.sample(range(total_stations), used_stations)
        combination.sort()  # Ensure combinations are sorted for uniqueness
        if combination not in random_combinations:
            random_combinations.append(combination)
    return random_combinations
#End of function


#generate sets of precipitation dataset with different gridded data coverage and different combination of grids coverage
grid = num_station
stn_keep_index = generate_random_combinations(len(stn), int(grid), num_combinations=total_comb)

for comb in np.arange(total_comb):
    #only keep the number of stations matching stn_keep_index
    stn_keep = stn.iloc[stn_keep_index[comb], :]

    #---HISTORICAL PRECIPITATION---#
    #only keep stations for this combination
    precip_keep = precip_all_historical[precip_all_historical['STATION'].isin(stn_keep['id'])].reset_index(drop=True)
    unique_dates = precip_keep['DATE'].unique()
    unique_dates = np.array([np.datetime64(date) for date in unique_dates])
    unique_dates = np.sort(unique_dates)
    #loop through each day, calculate inverse distance weighted precipitation for each grid point in the basin
    precip_df = pd.DataFrame() 
    for date in unique_dates:
        date = str(date)
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
        precip_mean = round(precip_mean, 4) #round to 4 digits
        precip_temp_df = pd.DataFrame({'DATE':date, 'PRECIP':[precip_mean]})
        precip_df = pd.concat([precip_df, precip_temp_df], ignore_index=True)
    #save the basin wise precipitation data to a csv file
    precip_df.to_csv(f'data/idw_precip/idw_precip{basin_id}_coverage{grid}_comb{comb}.csv', index=False)

    #---FUTURE PRECIPITATION---#
    #only keep stations for this combination
    precip_keep = precip_all_future[precip_all_future['STATION'].isin(stn_keep['id'])].reset_index(drop=True)
    #loop through each day, calculate inverse distance weighted precipitation for each grid point in the basin
    precip_df = pd.DataFrame() 
    unique_dates = precip_keep['DATE'].unique()
    unique_dates = np.array([np.datetime64(date) for date in unique_dates])
    unique_dates = np.sort(unique_dates)
    for date in unique_dates:
        date = str(date)
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
        precip_mean = round(precip_mean, 4) #round to 4 digits
        precip_temp_df = pd.DataFrame({'DATE':date, 'PRECIP':[precip_mean]})
        precip_df = pd.concat([precip_df, precip_temp_df], ignore_index=True)
    #save the basin wise precipitation data to a csv file
    precip_df.to_csv(f'data/future/future_idw_precip/future_idw_precip{basin_id}_coverage{grid}_comb{comb}.csv', index=False)