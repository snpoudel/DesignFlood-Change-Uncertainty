#load library
import numpy as np
import pandas as pd
import os

###---step 01---###
#  write objective functions for model evaluation ###
#write a function to calculate percentage high flow bias
def high_flow_bias(q_obs, q_sim):
    q_obs = np.array(q_obs)
    q_sim = np.array(q_sim)
    q_obs_995_value = np.percentile(q_obs, 99.9)
    indices_q995 = np.where(q_obs > q_obs_995_value)
    q_obs_995 = q_obs[indices_q995]
    q_sim_995 = q_sim[indices_q995]
    hfb = (np.sum(q_obs_995 - q_sim_995) / np.sum(q_obs_995)) * 100
    return hfb

#write a function to calculate RMSE
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value

###---step 02---###
basin_list = pd.read_csv("data/regional_lstm/MA_basins_gauges_2000-2020_filtered.csv", dtype={'basin_id':str})
used_basin_list = basin_list['basin_id']

df_total = pd.DataFrame() #create an empty dataframe to store the results
for id in used_basin_list:
    # id = '01109060'
    grid_coverage = np.arange(12)
    grid_coverage = np.append(grid_coverage, [99])

    #Historical
    #Loop through each basin
    for grid in grid_coverage: #1
        #read true streamflow
        true_hbv_flow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
        true_hbv_flow = true_hbv_flow[365:] #remove the first 365 days
        true_hbv_flow = true_hbv_flow.reset_index(drop=True)
        #read true precipitation
        true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')

        for combination in range(10): #2
            #Read real streamflow from interpolated precipitation
            file_path = f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{id}_coverage{grid}_comb{combination}.csv'
            if os.path.exists(file_path):
                #read real hbv flow
                # if os.path.exists(f'output/hbv_idw_streamflow/hbv_idw_streamflow{id}_coverage{grid}_comb{combination}.csv'):             
                real_hbv_flow = pd.read_csv(f'output/hbv_idw_streamflow/hbv_idw_streamflow{id}_coverage{grid}_comb{combination}.csv')
                real_hbv_flow = real_hbv_flow[365:] #remove the first 365 days
                real_hbv_flow = real_hbv_flow.reset_index(drop=True)
                #read recalibrated hbv flow
                recal_hbv_flow = pd.read_csv(f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{id}_coverage{grid}_comb{combination}.csv')
                recal_hbv_flow = recal_hbv_flow[365:] #remove the first 365 days
                recal_hbv_flow = recal_hbv_flow.reset_index(drop=True)
                #rea real hymod flow
                real_hymod_flow = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{id}_coverage{grid}_comb{combination}.csv')
                real_hymod_flow = real_hymod_flow[365:] #remove the first 365 days
                real_hymod_flow = real_hymod_flow.reset_index(drop=True)
                #read real lstm flow
                if os.path.exists(f'output/regional_lstm/historical/lstm_input{id}_coverage{grid}_comb{combination}.csv'):
                    real_lstm_flow = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage{grid}_comb{combination}.csv')
            
                #read real precipitation
                real_precip = pd.read_csv(f'data/idw_precip/idw_precip{id}_coverage{grid}_comb{combination}.csv')

                #now calculate nse, rmse, pbias, kge, hfb for real hbv and hymod streamflow against true hbv streamflow
                #calculate results only for validation period
                val_pd = 0 # ~ validation period starts from 2000th day
                #for hbv model

                hfb_hbv = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], real_hbv_flow['streamflow'][val_pd:])
                #for recalibrated hbv model
                hfb_recal_hbv = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], recal_hbv_flow['streamflow'][val_pd:])
                #for hymod model
                hfb_hymod = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], real_hymod_flow['streamflow'][val_pd:])
                #for lstm model
                if os.path.exists(f'output/regional_lstm/historical/lstm_input{id}_coverage{grid}_comb{combination}.csv'):
                    hfb_lstm = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], real_lstm_flow['streamflow'][val_pd:])
                else:
                    hfb_lstm = np.NAN

                #for precipitation
                rmse_precip = rmse(true_precip['PRECIP'], real_precip['PRECIP'])
                if np.array_equal(true_precip['PRECIP'], real_precip['PRECIP']):
                    rmse_precip = 0

                #save the results in a dataframe
                df_result = pd.DataFrame({'time':'historical', 'station_id':[id], 'grid':[grid], 'combination':[combination],
                                            'HFB(HBV)':[hfb_hbv],
                                            'HFB(RECAL_HBV)':[hfb_recal_hbv],
                                            'HFB(HYMOD)':[hfb_hymod],
                                            'HFB(LSTM)':[hfb_lstm],
                                            'RMSE(PRECIP)':[rmse_precip]})
                
                df_total = pd.concat([df_total, df_result], axis=0)


    #Future
    #Loop through each basin
    for grid in grid_coverage: #1
        #read true streamflow
        true_hbv_flow = pd.read_csv(f'output/future/hbv_true_future_streamflow/hbv_true_future_output_{id}.csv')
        true_hbv_flow = true_hbv_flow[365:] #remove the first 365 days
        true_hbv_flow = true_hbv_flow.reset_index(drop=True)
        #read true precipitation
        true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')

        for combination in range(10): #2
            #Read real streamflow from interpolated precipitation
            file_path = f'output/future/hbv_idw_recalib_future_streamflow/hbv_idw_recalib_future_streamflow{id}_coverage{grid}_comb{combination}.csv'
            if os.path.exists(file_path):
                #read real hbv flow
                # if os.path.exists(f'output/hbv_idw_streamflow/hbv_idw_streamflow{id}_coverage{grid}_comb{combination}.csv'):             
                real_hbv_flow = pd.read_csv(f'output/future/hbv_idw_future_streamflow/hbv_idw_future_streamflow{id}_coverage{grid}_comb{combination}.csv')
                real_hbv_flow = real_hbv_flow[365:] #remove the first 365 days
                real_hbv_flow = real_hbv_flow.reset_index(drop=True)
                #read recalibrated hbv flow
                recal_hbv_flow = pd.read_csv(f'output/future/hbv_idw_recalib_future_streamflow/hbv_idw_recalib_future_streamflow{id}_coverage{grid}_comb{combination}.csv')
                recal_hbv_flow = recal_hbv_flow[365:] #remove the first 365 days
                recal_hbv_flow = recal_hbv_flow.reset_index(drop=True)
                #rea real hymod flow
                real_hymod_flow = pd.read_csv(f'output/future/hymod_idw_future_streamflow/hymod_interpol_future_streamflow{id}_coverage{grid}_comb{combination}.csv')
                real_hymod_flow = real_hymod_flow[365:] #remove the first 365 days
                real_hymod_flow = real_hymod_flow.reset_index(drop=True)
                #read real lstm flow
                if os.path.exists(f'output/regional_lstm/future/lstm_input{id}_coverage{grid}_comb{combination}.csv'):
                    real_lstm_flow = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}_coverage{grid}_comb{combination}.csv')
            
                #read real precipitation
                real_precip = pd.read_csv(f'data/idw_precip/idw_precip{id}_coverage{grid}_comb{combination}.csv')

                #now calculate nse, rmse, pbias, kge, hfb for real hbv and hymod streamflow against true hbv streamflow
                #calculate results only for validation period
                val_pd = 0 # ~ validation period starts from 2000th day
                #for hbv model

                hfb_hbv = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], real_hbv_flow['streamflow'][val_pd:])
                #for recalibrated hbv model
                hfb_recal_hbv = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], recal_hbv_flow['streamflow'][val_pd:])
                #for hymod model
                hfb_hymod = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], real_hymod_flow['streamflow'][val_pd:])
                #for lstm model
                if os.path.exists(f'output/regional_lstm/historical/lstm_input{id}_coverage{grid}_comb{combination}.csv'):
                    hfb_lstm = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], real_lstm_flow['streamflow'][val_pd:])
                else:
                    hfb_lstm = np.NAN

                #for precipitation
                rmse_precip = rmse(true_precip['PRECIP'], real_precip['PRECIP'])
                if np.array_equal(true_precip['PRECIP'], real_precip['PRECIP']):
                    rmse_precip = 0

                #save the results in a dataframe
                df_result = pd.DataFrame({'time':'future', 'station_id':[id], 'grid':[grid], 'combination':[combination],
                                            'HFB(HBV)':[hfb_hbv],
                                            'HFB(RECAL_HBV)':[hfb_recal_hbv],
                                            'HFB(HYMOD)':[hfb_hymod],
                                            'HFB(LSTM)':[hfb_lstm],
                                            'RMSE(PRECIP)':[rmse_precip]})
                
                df_total = pd.concat([df_total, df_result], axis=0)
        #End of loop 23
        
df_total.to_csv(f'output/allbasins_diagnostics_entire_hist_future.csv', index=False)    
#End of loop 1
