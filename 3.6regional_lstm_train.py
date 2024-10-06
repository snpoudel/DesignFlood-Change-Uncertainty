#load libraries
import os
import pandas as pd
import numpy as np
from lstm_train import train_lstm_model
from mpi4py import MPI #to run parallel jobs on cluster

# ########
#Set up communicator to parallelize job in cluster using MPI
comm = MPI.COMM_WORLD #Get the default communicator object
rank = comm.Get_rank() #Get the rank of the current process
size = comm.Get_size() #Get the total number of processes

basin_list = pd.read_csv("data/regional_lstm/MA_basins_gauges_2000-2020_filtered.csv", dtype={'basin_id':str})
#precip buckets
precip_buckets = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
# precip_buckets = ['0', '0-1']
pb = 'pb' + precip_buckets[rank]

#merge lstm training datasets
df_train = pd.DataFrame()
for id in basin_list['basin_id']:
    if os.path.exists(f'data/regional_lstm/processed_lstm_input/{pb}/lstm_input{id}.csv'):
        temp_dataset = pd.read_csv(f'data/regional_lstm/processed_lstm_input/{pb}/lstm_input{id}.csv')
        temp_dataset = temp_dataset[0:5115]
        df_train = pd.concat([df_train, temp_dataset], ignore_index=True)
chunk_size_train = len(temp_dataset) #size of training dataset for single basin
lstm_train_dataset = df_train.drop(columns=['date'])
#train lstm model
train_lstm_model(lstm_train_dataset, chunk_size=chunk_size_train, precip_bucket = pb)

# Print overall GPU status using nvidia-smi
# os.system("nvidia-smi")