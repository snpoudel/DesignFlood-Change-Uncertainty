import numpy as np
import pandas as pd

'''
This script reads the csv files containing precipitation data for each station from 2000 to 2020,
replaces missing precipitation values with the mean precipitation value of the nearest neighbor 5 station for that day,
and saves the station wise precipitation data to a csv file.
'''

#read a list of all stations for MA 2000-2020
stn = pd.read_csv('data/MA_stations_2000-2020.csv')


precip_all_stations = pd.DataFrame()
#loop through each station
for id in stn['id']:
    #id= 'US1CTFR0009'
    precip_csv = pd.read_csv(f'data/gauge_precip_2000-2020/{id}.csv')
    precip_csv['DATE'] = pd.to_datetime(precip_csv['DATE'])
    #convert precipitation to mm
    precip_csv['PRCP'] = precip_csv['PRCP']/10
    #get the total number of days from 2000 to 2020
    total_days_2000_2020 = len(pd.date_range(start='2000-01-01', end='2020-12-31'))
    #if any of the days is missing in the data, add the missing days same station, lat, lon, and nan precip value
    if len(precip_csv) != total_days_2000_2020:
        #get the missing days
        missing_days = pd.date_range(start='2000-01-01', end='2020-12-31').difference(precip_csv['DATE'])
        #create a dataframe with missing days
        missing_days_df = pd.DataFrame({'DATE':missing_days, 'STATION':id,
                                        'LATITUDE':stn['lat'].values[0],
                                        'LONGITUDE':stn['lon'].values[0], 'PRCP':np.nan})
        #concatenate the missing days dataframe with the original dataframe
        precip_csv = pd.concat([precip_csv, missing_days_df], ignore_index=True)
    precip_all_stations = pd.concat([precip_all_stations, precip_csv], ignore_index=True)


#Replace missing precipitaiton value with precipitation value of nearest neighbor station for that day
#loop through each nan value in the PRCP column
for i in range(len(precip_all_stations)):
    #check if the value is nan
    if np.isnan(precip_all_stations['PRCP'][i]):
        #get the date of the nan value
        date_nan = precip_all_stations['DATE'][i]
        #get the station id
        station_id = precip_all_stations['STATION'][i]
        #get the latitude and longitude of the station
        lat = stn[stn['id'] == station_id]['lat'].values[0]
        lon = stn[stn['id'] == station_id]['lon'].values[0]
        #get the nearest neighbor station
        #calculate the distance between the station and all other stations
        stn['distance'] = np.sqrt((stn['lat'] - lat)**2 + (stn['lon'] - lon)**2)
        #sort the stations by distance
        stn = stn.sort_values(by='distance')
        #get the nearest neighbor station
        nearest_neighbor = stn.iloc[1:6]['id'] #use the 5 nearest neighbor stations
        #get the nearest neighbor station's precipitation data for the nan date from precip_all_stations
        precip_nn = precip_all_stations[precip_all_stations['STATION'].isin(nearest_neighbor)]
        #get the precipitation value of the nearest neighbor station for the date_nan
        precip_nn_value = precip_nn[precip_nn['DATE'] == date_nan]['PRCP']
        precip_nn_value = np.nanmean(precip_nn_value)
        #replace the nan value with the precipitation value of the nearest neighbor station
        precip_all_stations['PRCP'][i] = precip_nn_value
        

#save all precip as csv
precip_all_stations.to_csv('data/preprocessed_all_stations_precip_2000-2020.csv', index=False)
#save the station wise precipitation data from precip_all_stations to a csv file
for id in stn['id']:
    precip_station = precip_all_stations[precip_all_stations['STATION'] == id]
    precip_station.to_csv(f'data/preprocessed_gauge_precip_2000-2020/{id}.csv', index=False)