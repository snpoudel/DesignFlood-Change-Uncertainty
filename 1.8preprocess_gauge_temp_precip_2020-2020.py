import numpy as np
import pandas as pd

'''
This script reads the csv files containing precipitation, temperature data for each station from 2000 to 2020,
replaces missing precipitation and temp values with the mean precipitation value of the nearest neighbor 5 station for that day,
and saves the station wise precipitation data to a csv file.
'''

#read a list of all stations for MA 2000-2020
stn = pd.read_csv('data/swg_data/MA_gaugelist.csv')

precip_all_stations = pd.DataFrame()
#loop through each station
for id in stn['id']:
    #id= 'USC00191386'
    precip_csv = pd.read_csv(f'data/gauge_precip/{id}.csv', low_memory=False)
    temp_csv = pd.read_csv(f'data/gauge_temp/temp_{id}.csv')
    precip_csv['DATE'] = pd.to_datetime(precip_csv['DATE'])
    
    #only keep data from 2000-01-01 to 2020-12-31
    precip_csv = precip_csv[(precip_csv['DATE'] >= '2000-01-01') & (precip_csv['DATE'] <= '2020-12-31')]
    #only keep STATION, DATE, LATITUDE, LONGITUDE, PRCP
    precip_csv = precip_csv[['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'PRCP']]
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
        #sort the dataframe by date
        precip_csv = precip_csv.sort_values(by='DATE')
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
        lat_values = stn[stn['id'] == station_id]['lat'].values
        lon_values = stn[stn['id'] == station_id]['lon'].values
        if len(lat_values) > 0 and len(lon_values) > 0:
            lat = lat_values[0]
            lon = lon_values[0]
        else:
            continue
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
precip_all_stations.to_csv('data/swg_data/preprocessed_all_stations_precip_2000-2020.csv', index=False)
#save the station wise precipitation data from precip_all_stations to a csv file
for id in stn['id']:
    precip_station = precip_all_stations[precip_all_stations['STATION'] == id]
    precip_station.to_csv(f'data/swg_data/precip-gauges/{id}.csv', index=False)

#read in temperature data and merge with precipitation data
#temperature doezn't have missing values
for id in stn['id']:
    temp_csv = pd.read_csv(f'data/gauge_temp/temp_{id}.csv')
    temp_csv['time'] = pd.to_datetime(temp_csv['time'])
    temp_csv = temp_csv[(temp_csv['time'] >= '2000-01-01') & (temp_csv['time'] <= '2020-12-31')]
    #read in the corresponding precipitation data
    precip_csv = pd.read_csv(f'data/swg_data/precip-gauges/{id}.csv')
    precip_csv['DATE'] = pd.to_datetime(precip_csv['DATE'])
    #merge the temperature and precipitation data
    precip_csv = pd.merge(precip_csv, temp_csv, left_on='DATE', right_on='time', how='inner')
    precip_csv = precip_csv.drop(columns=['time'])
    #save the merged data to a csv file
    precip_csv.to_csv(f'data/swg_data/precip-temp-gauges/{id}.csv', index=False)
