#import libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import random
from mpi4py import MPI
from scipy.spatial import cKDTree

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Set up communicator to parallelize job in cluster using MPI
comm = MPI.COMM_WORLD #Get the default communicator object
rank = comm.Get_rank() #Get the rank of the current process
size = comm.Get_size() #Get the total number of processes

# Read basin lists with at least 2 stations
basin_list = pd.read_csv('data/ma29basins.csv', dtype={'basin_id': str})

# Prepare the basin list with necessary columns and combinations
basin_list['total_num_stn_used'] = basin_list['num_stations'] - 1
basin_list['stn_array'] = basin_list['total_num_stn_used'].apply(lambda x: np.arange(1, x + 1))
basin_list = basin_list.explode('stn_array').reset_index(drop=True)
basin_list['total_comb'] = basin_list['num_stations'].apply(lambda x: min(10, x))
basin_list['comb_array'] = basin_list['total_comb'].apply(lambda x: np.arange(1, x + 1))
basin_list = basin_list.explode('comb_array').reset_index(drop=True)

#extract necessary input for each MPI jobs
basin_id = basin_list['basin_id'][rank]
# total_comb = min(10, basin_list['num_stations'][rank])
num_station = basin_list['stn_array'][rank]
comb = basin_list['comb_array'][rank]

#read basin shapefile
basin_shapefile = gpd.read_file(f'data/prms_drainage_area_shapes/model_{basin_id}_nhru.shp')
#convert the basin shapefile to the same coordinate system as the stations
basin_shapefile = basin_shapefile.to_crs(epsg=4326)

#Divide the basin into meshgrid of size 0.125 degree
grid_size = 0.0625 #0.0625
if basin_id == '01108000': #for this basin, the grid size is 0.125
    grid_size = 0.125

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
def read_gauge_data(scenario):
    precip_all = pd.DataFrame()
    for gauge_id in gauge_stations['id']:
        gauge_data = pd.read_csv(f'data/swg_data/swg_output/processed/gauge_precip/{scenario}/{gauge_id}_scenario{scenario}.csv')
        gauge_data['STATION'] = gauge_id
        precip_all = pd.concat([precip_all, gauge_data], ignore_index=True)
    return precip_all

precip_all_historical = read_gauge_data(1)
precip_all_future = read_gauge_data(11)

#add the lat and lon of the stations to precip_all_stations from gauge_stations
precip_all_historical = pd.merge(precip_all_historical, gauge_stations[['id','lat','lon']], left_on='STATION', right_on='id', how='left') 
precip_all_future = pd.merge(precip_all_future, gauge_stations[['id','lat','lon']], left_on='STATION', right_on='id', how='left')

# Function to generate random combinations of grid cells
def generate_random_combinations(total_stations, used_stations, num_combinations): 
    random_combinations = []
    while len(random_combinations) < num_combinations:
        combination = random.sample(range(total_stations), used_stations)
        combination.sort()  # Ensure combinations are sorted for uniqueness
        if combination not in random_combinations:
            random_combinations.append(combination)
    return random_combinations

#generate sets of precipitation dataset with different gridded data coverage and different combination of grids coverage
grid = num_station
stn_keep_index = generate_random_combinations(len(gauge_stations), int(grid), num_combinations=comb)

# Create a KDTree for efficient distance calculations
tree = cKDTree(gdf_grid[['lon', 'lat']]) #ckdtree finds the nearest point in the grid to the station

def calculate_precipitation(precip_data, stn_keep, gdf_grid, tree):
    precip_df = pd.DataFrame()
    unique_dates = precip_data['date'].unique()
    for date in unique_dates:
        date = str(date)
        precip_day = precip_data[precip_data['date'] == date].reset_index(drop=True)
        precip_day = precip_day[precip_day['STATION'].isin(stn_keep['id'])].reset_index(drop=True)
        distances, indices = tree.query(precip_day[['lon', 'lat']], k=len(gdf_grid))
        weights = 1 / (distances ** 2)
        precip_weighted = np.sum(weights * precip_day['prcp'].values[:, np.newaxis], axis=0) / np.sum(weights, axis=0)
        precip_mean = np.mean(precip_weighted)
        precip_mean = round(precip_mean, 3)
        precip_temp_df = pd.DataFrame({'DATE': date, 'PRECIP': [precip_mean]})
        precip_df = pd.concat([precip_df, precip_temp_df], ignore_index=True)
    return precip_df

# Keep only the stations that are in the random combination
comb = comb - 1 # Subtract 1 to get the correct index
stn_keep = gauge_stations.iloc[stn_keep_index[comb], :]

# Historical precipitation
precip_df_historical = calculate_precipitation(precip_all_historical, stn_keep, gdf_grid, tree)
precip_df_historical.to_csv(f'data/noisy_precip/noisy_precip{basin_id}_coverage{grid}_comb{comb}.csv', index=False)

# Future precipitation
precip_df_future = calculate_precipitation(precip_all_future, stn_keep, gdf_grid, tree)
precip_df_future.to_csv(f'data/future/future_noisy_precip/future_noisy_precip{basin_id}_coverage{grid}_comb{comb}.csv', index=False)

print(f'Rank {rank} basin {basin_id} completed!!')

