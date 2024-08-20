#load libraries
import numpy as np
import pandas as pd


id = '01108000'
grid_coverage = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

#Loop through each basin
df_hbv_total = pd.DataFrame() #create an empty dataframe to store the results for HBV
df_hbv_recalib_total = pd.DataFrame() #create an empty dataframe to store the results for HBV recalibrated
df_hymod_total = pd.DataFrame() #create an empty dataframe to store the results for HYMOD
df_lstm_total = pd.DataFrame() #create an empty dataframe to store the results for LSTM
for grid in grid_coverage: #1
    true_flow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv') #read true hbv streamflow
    true_flow = true_flow[365:] #discard first 365 days to match with LSTM streamflow
    true_flow = true_flow.reset_index(drop=True)
    #only extract results for validation period
    val_pd = 5120 # ~ validation period starts from 12000th day
    true_flow = true_flow[val_pd:]
    #only extract rows from true_flow where streamflow is greater than 99.9th percentile
    true_flow = true_flow[true_flow['streamflow'] > true_flow['streamflow'].quantile(0.999)]
    for combination in range(10): #2
        #---HBV streamflow error calculation---#
        #Read real HBV streamflow
        real_hbv_flow = pd.read_csv(f'output/hbv_idw_streamflow/hbv_idw_streamflow{id}_coverage{grid}_comb{combination}.csv')
        real_hbv_flow = real_hbv_flow[real_hbv_flow['date'].isin(true_flow['date'] )] #only extract rows from real_hbv_flow where date is in true_flow

        #calculate difference in true and real HBV streamflow
        new_dataframe1 = pd.merge(real_hbv_flow, true_flow, on='date', how='inner')
        hbv_flow_error = new_dataframe1['streamflow_y'] - new_dataframe1['streamflow_x']
        hbv_flow_error = round(hbv_flow_error, 4)
        #store the HBV flow error in a dataframe
        hbv_flow_error_df = pd.DataFrame({'station_id': id, 'date':new_dataframe1['date'], 'grid': grid, 'combination': combination, 'flow_error': hbv_flow_error})  
        #combine this to the df_hbv_total dataframe
        df_hbv_total = pd.concat([df_hbv_total, hbv_flow_error_df], axis=0)

        #---HBV recalibrated streamflow error calculation---#
        #Read real HBV recalibrated streamflow
        real_hbv_recal_flow = pd.read_csv(f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{id}_coverage{grid}_comb{combination}.csv')
        real_hbv_recal_flow = real_hbv_recal_flow[real_hbv_recal_flow['date'].isin(true_flow['date'] )] #only extract rows from real_hbv_recal_flow where date is in true_flow
        #calculate difference in true and real HBV recalibrated streamflow
        new_dataframe4 = pd.merge(real_hbv_recal_flow, true_flow, on='date', how='inner')
        hbv_recal_flow_error = new_dataframe4['streamflow_y'] - new_dataframe4['streamflow_x']
        hbv_recal_flow_error = round(hbv_recal_flow_error, 4)
        #store the HBV recalibrated flow error in a dataframe
        hbv_recal_flow_error_df = pd.DataFrame({'station_id': id, 'date':new_dataframe4['date'], 'grid': grid, 'combination': combination, 'flow_error': hbv_recal_flow_error})
        #combine this to the df_hbv_total dataframe
        df_hbv_recalib_total = pd.concat([df_hbv_recalib_total, hbv_recal_flow_error_df], axis=0)

        #---HYMOD streamflow error calculation---#
        #Read real HYMOD streamflow
        real_hymod_flow = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{id}_coverage{grid}_comb{combination}.csv')
        real_hymod_flow = real_hymod_flow[real_hymod_flow['date'].isin(true_flow['date'] )] #only extract rows from real_hymod_flow where date is in true_flow
        #calculate difference in true and real HYMOD streamflow
        new_dataframe2 = pd.merge(real_hymod_flow, true_flow, on='date', how='inner')
        hymod_flow_error = new_dataframe2['streamflow_y'] - new_dataframe2['streamflow_x']
        hymod_flow_error = round(hymod_flow_error, 4)
        #store the HYMOD flow error in a dataframe
        hymod_flow_error_df = pd.DataFrame({'station_id': id, 'date':new_dataframe2['date'], 'grid': grid, 'combination': combination, 'flow_error': hymod_flow_error})
        #combine this to the df_hymod_total dataframe
        df_hymod_total = pd.concat([df_hymod_total, hymod_flow_error_df], axis=0)

        #---LSTM streamflow error calculation---#
        #Read real LSTM streamflow
        real_lstm_flow = pd.read_csv(f'output/lstm_idw_streamflow/lstm_idw_{id}_coverage{grid}_comb{combination}.csv')
        real_lstm_flow = real_lstm_flow[real_lstm_flow['date'].isin(true_flow['date'] )]
        #calculate difference in true and real LSTM streamflow
        new_dataframe3 = pd.merge(real_lstm_flow, true_flow, on = 'date', how = 'inner')
        lstm_flow_error = new_dataframe3['streamflow_y'] - new_dataframe3['streamflow_x']
        lstm_flow_error = round(lstm_flow_error, 4)
        #store the LSTM flow error in a dataframe
        lstm_flow_error_df = pd.DataFrame({'station_id': id, 'date':new_dataframe3['date'], 'grid': grid, 'combination': combination, 'flow_error': lstm_flow_error})
        #combine this to the df_lstm_total dataframe
        df_lstm_total = pd.concat([df_lstm_total, lstm_flow_error_df], axis=0)
    #End of loop #2
    
df_hbv_total.to_csv('output/99.9hbvflow_error_distribution_validperiod.csv', index=False)
df_hbv_recalib_total.to_csv('output/99.9hbv_recalibflow_error_distribution_validperiod.csv', index=False)
df_hymod_total.to_csv('output/99.9hymodflow_error_distribution_validperiod.csv', index=False)  
df_lstm_total.to_csv('output/99.9lstmflow_error_distribution_validperiod.csv', index=False)
#End of loop #1

