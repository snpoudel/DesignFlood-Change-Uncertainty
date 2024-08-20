#load libraries
import os
import gc #this library is used to free ram memory
import pandas as pd
import numpy as np
from lstm_model_cpu import run_lstm_model
from mpi4py import MPI #to run parallel jobs on cluster

########
#Set up communicator to parallelize job in cluster using MPI
comm = MPI.COMM_WORLD #Get the default communicator object
rank = comm.Get_rank() #Get the rank of the current process
size = comm.Get_size() #Get the total number of processes

#read the list of basin ID with centriod latitude
lat_basin = pd.read_csv('data/basinID_withLatLon.csv', dtype={'STAID':str})
# Select one basin from the list/ need to do a for loop if not running parallely in cluster
id = '01108000'
#generate sets of precipitation dataset with different gridded data coverage and different combinatoin of grids coverage
station_coverage = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
grid = station_coverage[rank] #select grid coverage based on rank
########### Simulate streamflow for a calibrated hymod model ###########
for combination in range(10): #1
    #Read interpolated precipitation
    file_path = f'data/idw_precip/idw_precip{id}_coverage{grid}_comb{combination}.csv'
    if os.path.exists(file_path):
        precip_in = pd.read_csv(file_path)
        #Read temperature era5
        temp_in = pd.read_csv(f'data/processed-era5-temp/temp_{id}.csv')
        #filter temperature for the year 2000-2020
        temp_in = temp_in[temp_in['time'] >= '2000-01-01']
        temp_in = temp_in[temp_in['time'] <= '2020-12-31']
        temp_in = temp_in.reset_index(drop=True)
        #Read latitude
        lat_in_df = lat_basin[lat_basin['STAID'] == id]
        lat_in = lat_in_df['LAT_CENT'].iloc[0]

        #Read observed streamflow
        q_obs = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
        
        #train lstm model
        #prepare lstm dataset
        lstm_dataset = pd.DataFrame({'precip':precip_in['PRECIP'], 'temp':temp_in['t2m'],
                                          'latitude':lat_in, 'qobs':q_obs['streamflow']})
        
        #train lstm model and make predictions
        q_sim = run_lstm_model(lstm_dataset)
        q_sim = np.round(q_sim, 4)
        q_sim = q_sim.flatten()
        #calculate nse between q_obs and q_sim
        #nse = 1 - (np.sum((q_obs['streamflow'][365:] - q_sim)**2) / np.sum((q_obs['streamflow'][365:] - np.mean(q_obs['streamflow'][365:]))**2))
        #keep result in a dataframe
        output_df = pd.DataFrame({ 'date':precip_in['DATE'][365:], 'streamflow':q_sim }) #the first 365 days are used as sequence length for lstm model
        #save output dataframe
        output_df.to_csv(f'output/lstm_idw_streamflow/lstm_idw_{id}_coverage{grid}_comb{combination}.csv')
        #clear unused variables and free memory before next iteration
        del precip_in, temp_in, lat_in_df, lat_in, q_obs, lstm_dataset, q_sim, output_df
        #free ram memory after each iteration
        gc.collect() #garbage collection to free ram memory
        

#End of Loop 1