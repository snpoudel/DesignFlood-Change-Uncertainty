#load libraries
import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from lstm_model_gpu import run_lstm_model
from mpi4py import MPI #to run parallel jobs on cluster

########
#Set up communicator to parallelize job in cluster using MPI
comm = MPI.COMM_WORLD #Get the default communicator object
rank = comm.Get_rank() #Get the rank of the current process
size = comm.Get_size() #Get the total number of processes

#precip buckets
precip_buckets = ['0-2', '2-4', '4-6', '6-8', '8-10']
precip_buckets = ['8-10']
#precip_buckets = precip_buckets[rank]
for pb in precip_buckets:
    file_path = f'data/regional_lstm/processed_lstm_input/pb{pb}/*.csv'

    csv_file_paths = glob.glob(file_path) #file path
    csv_file_names = [os.path.basename(file) for file in csv_file_paths] #file name


    df = pd.DataFrame() #empty dataframe 
    df_train = pd.DataFrame() 
    df_test = pd.DataFrame()
    for file in csv_file_paths:
        temp = pd.read_csv(file)
        temp_train, temp_test = train_test_split(temp, test_size=0.9) #0.35
        df = pd.concat([df, temp], ignore_index=True)
        df_train = pd.concat([df_train, temp_train], ignore_index=True)
        df_test = pd.concat([df_test, temp_test], ignore_index=True)

    #--HISTORICAL--#
    #prepare historical lstm dataset 
    date_lstm_dataset = df['date']
    date_lstm_dataset = date_lstm_dataset[365:]

    lstm_dataset = df.drop(columns=['date'])
    lstm_train_dataset = df_train.drop(columns=['date'])
    lstm_test_dataset = df_test.drop(columns=['date'])

    chunk_size_total = len(temp) #size of dataset for single basin
    chunk_size_train = len(temp_train) #size of training dataset for single basin


    #--FUTURE--#
    #prepare future lstm dataset
    file_path = f'data/regional_lstm/future_processed_lstm_input/pb{pb}/*.csv'

    csv_file_paths = glob.glob(file_path) #file path
    csv_file_names = [os.path.basename(file) for file in csv_file_paths] #file name

    df = pd.DataFrame() #empty dataframe 
    for file in csv_file_paths:
        temp = pd.read_csv(file)
        df = pd.concat([df, temp], ignore_index=True)

    lstm_future_dataset = df.drop(columns=['date'])


    #train lstm model and make predictions
    q_sim, q_sim_future = run_lstm_model(lstm_dataset, lstm_train_dataset,
                                          lstm_test_dataset, lstm_future_dataset, chunk_size=chunk_size_train)

    q_sim = np.round(q_sim, 4)
    q_sim = q_sim.flatten()
    q_sim_future = np.round(q_sim_future, 4)
    q_sim_future = q_sim_future.flatten()

    #save output
    i = 0
    for index, file_name in enumerate(csv_file_names):
        output_df = pd.DataFrame({'date':date_lstm_dataset[i:i+chunk_size_total], 'streamflow':q_sim[i:i+chunk_size_total]})
        future_output_df = pd.DataFrame({'date':date_lstm_dataset[i:i+chunk_size_total], 'streamflow':q_sim_future[i:i+chunk_size_total]})
        #save output dataframe
        output_df.to_csv(f'output/regional_lstm/historical/{csv_file_names[index]}')
        future_output_df.to_csv(f'output/regional_lstm/future/{csv_file_names[index]}')
        i+=chunk_size_total