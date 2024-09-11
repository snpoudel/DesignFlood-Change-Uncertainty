#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

#read MA stations that has precipitation data starting from 2010 upto 2020
station_MA = pd.read_csv('data/MA_stations_2000-2020.csv')
#read basin id
df_basin_id = pd.read_csv('data/basin_id_withshapefile_new.csv',dtype={'station_id':str})
#empty dataframe to store the number of stations in each basin
df_num_stations = pd.DataFrame()
#read a basin shapefile

for basin_id in df_basin_id['station_id']:
    # basin_id = '01108000'
    basin_shapefile = gpd.read_file(f'data/prms_drainage_area_shapes/model_{basin_id}_nhru.shp')
    #create another shapefile name basin_shapefile_buffer with 10km buffer
    basin_shapefile_buffer = basin_shapefile.copy()
    basin_shapefile_buffer['geometry'] = basin_shapefile_buffer.buffer(10000) #buffer 10km

    #convert the basin shapefile to the same coordinate system as the stations
    basin_shapefile = basin_shapefile.to_crs(epsg=4326)
    basin_shapefile_buffer = basin_shapefile_buffer.to_crs(epsg=4326)
    #filter stations that lies within the basin_shapefile
    #convert the stations to geodataframe
    stn_MA = gpd.GeoDataFrame(station_MA,geometry=gpd.points_from_xy(station_MA['lon'],station_MA['lat']))
    #make a list of stations that overlap with the basin
    stn_MA = stn_MA[stn_MA.within(basin_shapefile_buffer.unary_union)]
    stn_MA = stn_MA.reset_index(drop=True)

    # ###plot the stations and the basin
    # fig, ax = plt.subplots(figsize=(5,5))
    # basin_shapefile.plot(ax=ax)
    # stn_MA.plot(ax=ax,color='red', markersize=50, label='Gauging Station', marker='x')
    # # plt.xlim([-71.6,-71.1])
    # #plt.ylim([41.9,42.5])
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.legend(loc = 'lower left')
    # plt.title(f'Precip Gauging Stations for Basin: {basin_id}')
    # plt.tight_layout()
    # plt.show()
    # #save figure
    # fig.savefig(f'output/basin_{basin_id}.png', dpi=300)

    #save a csv file that contains the station id and the number of stations
    num_stations = len(stn_MA)
    temp_df = pd.DataFrame({'basin_id':[basin_id],'num_stations':[num_stations]})
    df_num_stations = pd.concat([df_num_stations, temp_df], ignore_index=True)
    #save the basin with stations as a csv file
    stn_MA.to_csv(f'data/num_gauge_precip_basinwise_2000-2020/basin_{basin_id}.csv',index=False)

df_num_stations.to_csv('data/MA_gauges_per_basin_2000-2020.csv',index=False)
