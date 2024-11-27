import pandas as pd
import os
import pyreadr

# Step 1: Combine all CSV files
data_dir = "data/swg_data/precip-temp-gauges"  # Directory where the  CSV files are stored
file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".csv")]

# Load all CSV files into a single DataFrame
all_data = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)
#keep lattitude and longitude to three decimal places
all_data['LATITUDE'] = all_data['LATITUDE'].round(3)
all_data['LONGITUDE'] = all_data['LONGITUDE'].round(3)
#rename t2m as TAVG
all_data = all_data.rename(columns={'t2m':'TAVG'})

# Step 2: Create a MultiIndex DataFrame with rows as DATE and columns as (LAT, LON)
gauge_list = pd.read_csv('data/swg_data/MA_gaugelist.csv')
gauge_list['lat'] = gauge_list['lat'].round(3)
gauge_list['lon'] = gauge_list['lon'].round(3)
#extract lat_lon as lat/lon
gauge_list['lat_lon'] = gauge_list['lat'].astype(str) + '/' + gauge_list['lon'].astype(str)
lat_lon = gauge_list['lat_lon'].values

#pivot for prcp by station
prcp_pivot = all_data.pivot_table(index='DATE', columns='STATION', values='PRCP')

#pivot for tavg by station
tavg_pivot = all_data.pivot_table(index='DATE', columns='STATION', values='TAVG')

#change column names from station id to lat/lon
prcp_pivot.columns = lat_lon
tavg_pivot.columns = lat_lon

#save the pivoted dataframes as RDS files
pyreadr.write_rds("data/swg_data/precip_pivot.rds", prcp_pivot)
pyreadr.write_rds("data/swg_data/tavg_pivot.rds", tavg_pivot)