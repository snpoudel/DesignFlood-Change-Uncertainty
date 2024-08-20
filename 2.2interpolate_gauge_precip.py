#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import random
import multiprocessing
#basid id
basin_id = '01108000'
#read basin shapefile
basin_shapefile = gpd.read_file(f'data/prms_drainage_area_shapes/model_{basin_id}_nhru.shp')
#convert the basin shapefile to the same coordinate system as the stations
basin_shapefile = basin_shapefile.to_crs(epsg=4326)

##Divide the basin into meshgrid of size 0.125 degree
#find the bounding box of the basin
grid_size = 0.03125
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

# Function to generate random combinations of grid cells
# This avoids use of combinations from itertools, as it is slow for large number of grid cells
def generate_random_combinations(total_stations, used_stations, num_combinations=10): 
    random_combinations = []
    while len(random_combinations) < num_combinations:
        combination = random.sample(range(total_stations), used_stations)
        combination.sort()  # Ensure combinations are sorted for uniqueness
        if combination not in random_combinations:
            random_combinations.append(combination)
    return random_combinations
#End of function

#grid coverage
# station_coverage = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# possible_num_of_station = np.ceil(len(stn) * station_coverage)
# possible_num_of_station
#multiprocessing function
def func_multiprocessing(grid_input):
        num_stn_coverage = np.ceil(len(stn) * grid_input)
        stn_keep_index = generate_random_combinations(len(stn), int(num_stn_coverage))
        #loop through each combination
        for comb in range(10):
            #only keep the number of stations matching stn_keep_index
            stn_keep = stn.iloc[stn_keep_index[comb], :]
            #Read all precip data
            precip_all_stations = pd.read_csv('data/preprocessed_all_stations_precip_2000-2020.csv')
            #only keep stations that are within the basin
            precip_keep = precip_all_stations[precip_all_stations['STATION'].isin(stn_keep['id'])].reset_index(drop=True)

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
                precip_mean = round(precip_mean, 4) #round to 4 digits
                precip_temp_df = pd.DataFrame({'DATE':date, 'PRECIP':[precip_mean]})
                precip_df = pd.concat([precip_df, precip_temp_df], ignore_index=True)

            #save the basin wise precipitation data to a csv file
            precip_df.to_csv(f'data/idw_precip/idw_precip{basin_id}_coverage{grid_input}_comb{comb}.csv', index=False)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=10)
    grid_input = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    pool.map(func_multiprocessing, grid_input)
    pool.close()
    pool.join()
