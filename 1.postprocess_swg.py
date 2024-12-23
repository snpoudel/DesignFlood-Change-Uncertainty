import numpy as np
import pandas as pd
import os

#read simulated dates
sim_date = pd.read_csv('data/swg_data/swg_output/sim.dates.csv')
#merge year month day columns to make date as string
sim_date['date'] = sim_date['year'].astype(str) + '-' + sim_date['month'].astype(str) + '-' + sim_date['day'].astype(str)
date_list = sim_date['date']

#read basin list
basin_list = pd.read_csv('data/swg_data/MA_gaugelist_withbasins_cat.csv', dtype={'basin_id':str})
basin_list['lat/lon'] = basin_list['lat'].astype(str) + '/' + basin_list['lon'].astype(str)
#make list of simulated dates for 100 years

#1:base, 3:temp2,precip0,cc3.5, 7:temp2,precip10,cc3.5
#11:temp2,precip0,cc7.0, 15:temp2,precip10,cc7.0
scenario_list = [1,3,7,11,15] #base and climate change scenarios of interest
for scenario in scenario_list:
    #make two lists which includes all csv files that starts with 'tmean' and 'prcp'
    tmean_files = [f for f in os.listdir(f'data/swg_data/swg_output/scenarios_csv/{scenario}') if f.startswith(f'tmean')]
    prcp_files = [f for f in os.listdir(f'data/swg_data/swg_output/scenarios_csv/{scenario}') if f.startswith(f'prcp')]

    #loop through each temp_files
    for tmean_file in tmean_files:
        #read the tmean file
        tmean = pd.read_csv(f'data/swg_data/swg_output/scenarios_csv/{scenario}/{tmean_file}').iloc[:len(date_list)]
        tmean.insert(0,'date',date_list) #insert date column
        tmean.columns = tmean.columns.str.strip()#remove whitespace from column names
        #rename column names from lat/lon to id using basin_list and save the dataframe
        for col in tmean.columns[1:]: #loop through each column except date
            #convert col to string
            temp_df = tmean[['date',col]]
            #find id name for the column col
            id_name = basin_list[basin_list['lat/lon'] == col]['id'].values[0]
            temp_df = temp_df.rename(columns={col:'tavg'})
            #save the dataframe
            temp_df.to_csv(f'data/swg_data/swg_output/processed/gauge_temp/{scenario}/{id_name}_scenario{scenario}.csv',index=False)

    #loop through each prcp_files
    for prcp_file in prcp_files:
        #read the prcp file
        prcp = pd.read_csv(f'data/swg_data/swg_output/scenarios_csv/{scenario}/{prcp_file}').iloc[:len(date_list)]
        prcp.insert(0,'date',date_list) #insert date column
        prcp.columns = prcp.columns.str.strip()#remove whitespace from column names
        #rename column names from lat/lon to id using basin_list and save the dataframe
        for col in prcp.columns[1:]: #loop through each column except date
            #convert col to string
            temp_df = prcp[['date',col]]
            #find id name for the column col
            id_name = basin_list[basin_list['lat/lon'] == col]['id'].values[0]
            temp_df.rename(columns={col:'prcp'})
            #save the dataframe
            temp_df.to_csv(f'data/swg_data/swg_output/processed/gauge_precip/{scenario}/{id_name}_scenario{scenario}.csv',index=False)