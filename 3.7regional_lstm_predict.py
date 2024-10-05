#load libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lstm_predict import predict_lstm_model
from mpi4py import MPI #to run parallel jobs on cluster

# ########
# #Set up communicator to parallelize job in cluster using MPI
comm = MPI.COMM_WORLD #Get the default communicator object
rank = comm.Get_rank() #Get the rank of the current process
size = comm.Get_size() #Get the total number of processes

basin_list = pd.read_csv("data/regional_lstm/MA_basins_gauges_2000-2020_filtered.csv", dtype={'basin_id':str})
#precip buckets
precip_buckets = ['1-2', '3-4', '4-6']
pb = 'pb' + precip_buckets[rank]


#make prediction for historical dataset
df_predict = pd.DataFrame()

for id in basin_list['basin_id']:
    if os.path.exists(f'data/regional_lstm/processed_lstm_input/{pb}/lstm_input{id}.csv'):
        temp_dataset = pd.read_csv(f'data/regional_lstm/processed_lstm_input/{pb}/lstm_input{id}.csv')
        temp_dataset = temp_dataset[0:5115]
        df_predict = pd.concat([df_predict, temp_dataset], ignore_index=True)
df_predict = df_predict.drop(columns=['date'])

features = df_predict.iloc[:, :-1].values # 2d array of features, everything is features exccept last column
#normalize features
scaler = StandardScaler() #standardize features by removing the mean and scaling to unit variance
features = scaler.fit_transform(features) #fit to data, then transform it

#Historical
#read basin for which prediction is to be made, and normalize features using scaler fitted to training data 
for id in basin_list['basin_id']:
    for coverage in np.arange(10):
        for comb in np.arange(10):
            file_path = f'data/regional_lstm/prediction_datasets/historical/{pb}/lstm_input{id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                temp_dataset = pd.read_csv(file_path)
                temp_date = temp_dataset['date'][365:]
                temp_dataset = temp_dataset.drop(columns=['date'])
                temp_features = df_predict.iloc[:, :-1].values
                #scale
                temp_features = scaler.transform(temp_features)
                q_sim = predict_lstm_model(temp_features, precip_bucket = pb)
                q_sim = np.round(q_sim, 3)
                temp_df = pd.DataFrame({'date':temp_date, 'streamflow':q_sim})
                temp_df.to_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage{coverage}_comb{comb}.csv', index=False)


#Future
#read basin for which prediction is to be made, and normalize features using scaler fitted to training data 
for id in basin_list['basin_id']:
    for coverage in np.arange(10):
        for comb in np.arange(10):
            file_path = f'data/regional_lstm/prediction_datasets/future/{pb}/lstm_input{id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                temp_dataset = pd.read_csv(file_path)
                temp_date = temp_dataset['date'][365:]
                temp_dataset = temp_dataset.drop(columns=['date'])
                temp_features = df_predict.iloc[:, :-1].values
                #scale
                temp_features = scaler.transform(temp_features)
                q_sim = predict_lstm_model(temp_features, precip_bucket = pb)
                q_sim = np.round(q_sim, 3)
                temp_df = pd.DataFrame({'date':temp_date, 'streamflow':q_sim})
                temp_df.to_csv(f'output/regional_lstm/future/lstm_input{id}_coverage{coverage}_comb{comb}.csv', index=False)

