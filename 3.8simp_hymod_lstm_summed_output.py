#load libraries
import numpy as np
import pandas as pd
import os

#read basin list
basin_list = pd.read_csv("data/regional_lstm/MA_basins_gauges_2000-2020_filtered.csv", dtype={'basin_id':str})['basin_id'].values

#Historical
for id in basin_list:
    for coverage in range(105):
        for comb in range(15):
            file_path = f'output/regional_lstm_simp_hymod/historical/lstm_input{id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                lstm_error = pd.read_csv(file_path)
                hymod_simflow = pd.read_csv(f'output/simp-hymod/hymod_interpol_streamflow{id}_coverage{coverage}_comb{comb}.csv')
                #merge by date
                temp_df = pd.merge(lstm_error, hymod_simflow, on='date', how='inner')
                temp_df['hymod_lstm_streamflow']=temp_df['streamflow']+temp_df['streamflow_error']

                #only extract date and hymod_lstm_streamflow
                temp_df = temp_df.loc[:,['date', 'hymod_lstm_streamflow']]
                #only keep date starting from 2000-12-30
                temp_df = temp_df[temp_df['date']>='2000-12-30']
                #save as a csv file
                temp_df.to_csv(f'output/regional_lstm_simp_hymod/final_output/historical/hymod_lstm{id}_coverage{coverage}_comb{comb}.csv')

#Future
for id in basin_list:
    for coverage in range(105):
        for comb in range(15):
            file_path = f'output/regional_lstm_simp_hymod/future/lstm_input{id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                lstm_error = pd.read_csv(file_path)
                hymod_simflow = pd.read_csv(f'output/future/simp-hymod/hymod_interpol_future_streamflow{id}_coverage{coverage}_comb{comb}.csv')
                #merge by date
                temp_df = pd.merge(lstm_error, hymod_simflow, on='date', how='inner')
                temp_df['hymod_lstm_streamflow']=temp_df['streamflow']+temp_df['streamflow_error']

                #only extract date and hymod_lstm_streamflow
                temp_df = temp_df.loc[:,['date', 'hymod_lstm_streamflow']]
                #only keep date starting from 2000-12-30
                temp_df = temp_df[temp_df['date']>='2000-12-30']

                #save as a csv file
                temp_df.to_csv(f'output/regional_lstm_simp_hymod/final_output/future/hymod_lstm{id}_coverage{coverage}_comb{comb}.csv')
                