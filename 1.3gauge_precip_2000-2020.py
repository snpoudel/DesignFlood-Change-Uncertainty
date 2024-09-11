#this script keeps gauging station that has precipitation data starting from 2000 upto 2020
# import libraries
import pandas as pd
import numpy as np
import os

stn = pd.read_csv('data/MA_stations.csv')
#remove rows containing these stations, as they have some problem
remove_stn = ['USR0000RNIN','US1CTHR0085','US1RIKN0034']
stn = stn[~stn['id'].isin(remove_stn)] # ~ is a logical not operator

stn_2000_2020 = [] #list to store stations that have precipitation data starting from 2000 to 2020
#loop through each gauging station
for id in stn['id']:
    #id = 'US1MAMD0005'
    #read the csv file
    csv_path = f'data/gauge_precip/{id}.csv'
    if os.path.exists(csv_path): #check if the csv file exists
        df = pd.read_csv(f'data/gauge_precip/{id}.csv')
        #convert the date column to date time type
        df['DATE'] = pd.to_datetime(df['DATE'])
        #extract the year from the date
        df['YEAR'] = df['DATE'].dt.year
        #only keep the date
        df['DATE'] = df['DATE'].dt.date
        #filter the csv file to only keep precipitation data starting from 2000 to 2020
        df = df[df['YEAR'] >= 2000]
        df = df[df['YEAR'] <= 2020]
        #total days of data available from 2000 to 2020
        total_days = len(np.arange('2000-01-01','2021-01-01',dtype='datetime64[D]'))
        total_days_data = len(df['DATE'].unique())
        #unique years in the year column
        df_unique_years = df['YEAR'].unique()
        #check if df year column completely contains the range of 2000 to 2020
        if len(df_unique_years) == len(np.arange(2000,2021)): #if the length of the unique years is equal to the length of the range of 2000 to 2020
            if total_days_data > 0.95*total_days: #if the total days of data available is greater than 99% of the total days from 2000 to 2020
                #save the csv file to a folder
                #filter the csv file to only keep precipitation data starting from 2000
                df = df[df['YEAR'] >= 2000]
                #only keep the first 7 columns
                col_names = ['STATION', 'DATE', 'LATITUDE','LONGITUDE','PRCP']
                df = df[col_names]
                #save the csv file
                df.to_csv(f'data/gauge_precip_2000-2020/{id}.csv', index=False)
                #append the station id to the list
                stn_2000_2020.append(id) #append the station id to the list
        #end of if statement
#end of for loop    

#filter rows from MA_stations.csv that are in stn_2000_2020
df_stn_2000_2020 = stn[stn['id'].isin(stn_2000_2020)]
#save the filtered rows to a csv file
df_stn_2000_2020.to_csv('data/MA_stations_2000-2020.csv', index=False)

