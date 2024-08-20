#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
#read all stations txt file
stn = pd.read_table('data/ghcnd-stations.txt')
# seperate columns by space
stn = stn.iloc[:,0].str.split(expand=True)
#write column names for first 4 and discard the rest
stn = stn.iloc[:,0:4]
stn.columns = ['id','lat','lon','elev']

#filter stations in MA based on latitude and longitude
#change the lat and lon to float
stn['lat'] = stn['lat'].astype(float)
stn['lon'] = stn['lon'].astype(float)

#filter stations in MA
stn_MA = stn[(stn['lat']>=41.2) & (stn['lat']<=42.9) & (stn['lon']>=-73.5) & (stn['lon']<=-69.9)]
#save the stations in MA as a csv file
stn_MA.to_csv('data/MA_stations.csv',index=False)
#read a basin shapefile
basin_shapefile = gpd.read_file('data/prms_drainage_area_shapes/model_01104500_nhru.shp')
#plot the basin shapefile
#convert the basin shapefile to the same coordinate system as the stations
basin_shapefile = basin_shapefile.to_crs(epsg=4326)
#filter stations that lies within the basin_shapefile
#convert the stations to geodataframe
stn_MA = gpd.GeoDataFrame(stn_MA,geometry=gpd.points_from_xy(stn_MA['lon'],stn_MA['lat']))
#make a list of stations that overlap with the basin
stn_MA = stn_MA[stn_MA.within(basin_shapefile.unary_union)]
# #plot the stations and the basin
# fig, ax = plt.subplots()
# basin_shapefile.plot(ax=ax)
# stn_MA.plot(ax=ax,color='red', markersize=25, label='Station', marker='x')
# plt.xlim([-71.6,-71.1])
# plt.ylim([42,42.5])
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.show()

#save the basin with stations as a csv file
stn_MA.to_csv('data/basin-num_stations/basin_01104500_stations.csv',index=False)


