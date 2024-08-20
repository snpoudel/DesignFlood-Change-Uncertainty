#load library
import numpy as np
import pandas as pd
import os

###---step 01---###
#  write objective functions for model evaluation ###
#write a function to calculate NSE
def nse(q_obs, q_sim):
    numerator = np.sum((q_obs - q_sim)**2)
    denominator = np.sum((q_obs - (np.mean(q_obs)))**2)
    nse_value = 1 - (numerator/denominator)
    return nse_value

#write a function to calculate RMSE
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value

#write a function to calculate Percentage BIAS
def pbias(q_obs, q_sim):
    pbias_value = (np.sum(q_obs - q_sim) / np.sum(q_obs)) * 100
    return pbias_value

#write a function to calculate KGE
def kge(q_obs, q_sim):
    r = np.corrcoef(q_obs, q_sim)[0,1]
    alpha = np.std(q_sim) / np.std(q_obs)
    beta = np.mean(q_sim) / np.mean(q_obs)
    kge_value = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    return kge_value

#write a function to calculate percentage high flow bias
def high_flow_bias(q_obs, q_sim):
    q_obs = np.array(q_obs)
    q_sim = np.array(q_sim)
    q_obs_995 = np.percentile(q_obs, 99.9)
    indices_q995 = np.where(q_obs > q_obs_995)
    q_sim_995 = q_sim[indices_q995]
    hfb = (np.sum(q_obs_995 - q_sim_995) / np.sum(q_obs_995)) * 100
    return hfb


###---step 02---###
id = '01108000'
grid_coverage = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

#Loop through each basin
df_total = pd.DataFrame() #create an empty dataframe to store the results
for grid in grid_coverage: #1
    #read true streamflow
    true_hbv_flow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
    true_hbv_flow = true_hbv_flow[365:] #remove the first 365 days
    true_hbv_flow = true_hbv_flow.reset_index(drop=True)
    #read true precipitation
    true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')
    true_precip = true_precip[365:] #remove the first 365 days
    true_precip = true_precip.reset_index(drop=True)

    for combination in range(10): #2
        #Read real streamflow from interpolated precipitation
        file_path = f'output/hbv_idw_streamflow/hbv_idw_streamflow{id}_coverage{grid}_comb{combination}.csv'
        if os.path.exists(file_path):
            #read real hbv flow
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
            real_lstm_flow = pd.read_csv(f'output/lstm_idw_streamflow/lstm_idw_{id}_coverage{grid}_comb{combination}.csv')
            #read real precipitation
            real_precip = pd.read_csv(f'data/idw_precip/idw_precip{id}_coverage{grid}_comb{combination}.csv')
            real_precip = real_precip[365:] #remove the first 365 days
            real_precip = real_precip.reset_index(drop=True)

            #now calculate nse, rmse, pbias, kge, hfb for real hbv and hymod streamflow against true hbv streamflow
            #calculate results only for validation period
            val_pd = 5120 # ~ validation period starts from 2000th day
            #for hbv model
            nse_hbv = nse(true_hbv_flow['streamflow'][val_pd:], real_hbv_flow['streamflow'][val_pd:])
            rmse_hbv = rmse(true_hbv_flow['streamflow'][val_pd:], real_hbv_flow['streamflow'][val_pd:])
            pbias_hbv = pbias(true_hbv_flow['streamflow'][val_pd:], real_hbv_flow['streamflow'][val_pd:])
            kge_hbv = kge(true_hbv_flow['streamflow'][val_pd:], real_hbv_flow['streamflow'][val_pd:])
            hfb_hbv = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], real_hbv_flow['streamflow'][val_pd:])
            #for recalibrated hbv model
            nse_recal_hbv = nse(true_hbv_flow['streamflow'][val_pd:], recal_hbv_flow['streamflow'][val_pd:])
            rmse_recal_hbv = rmse(true_hbv_flow['streamflow'][val_pd:], recal_hbv_flow['streamflow'][val_pd:])
            pbias_recal_hbv = pbias(true_hbv_flow['streamflow'][val_pd:], recal_hbv_flow['streamflow'][val_pd:])
            kge_recal_hbv = kge(true_hbv_flow['streamflow'][val_pd:], recal_hbv_flow['streamflow'][val_pd:])
            hfb_recal_hbv = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], recal_hbv_flow['streamflow'][val_pd:])
            #for hymod model
            nse_hymod = nse(true_hbv_flow['streamflow'][val_pd:], real_hymod_flow['streamflow'][val_pd:])
            rmse_hymod = rmse(true_hbv_flow['streamflow'][val_pd:], real_hymod_flow['streamflow'][val_pd:])
            pbias_hymod = pbias(true_hbv_flow['streamflow'][val_pd:], real_hymod_flow['streamflow'][val_pd:])
            kge_hymod = kge(true_hbv_flow['streamflow'][val_pd:], real_hymod_flow['streamflow'][val_pd:])
            hfb_hymod = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], real_hymod_flow['streamflow'][val_pd:])
            #for lstm model
            nse_lstm = nse(true_hbv_flow['streamflow'][val_pd:], real_lstm_flow['streamflow'][val_pd:])
            rmse_lstm = rmse(true_hbv_flow['streamflow'][val_pd:], real_lstm_flow['streamflow'][val_pd:])
            pbias_lstm = pbias(true_hbv_flow['streamflow'][val_pd:], real_lstm_flow['streamflow'][val_pd:])
            kge_lstm = kge(true_hbv_flow['streamflow'][val_pd:], real_lstm_flow['streamflow'][val_pd:])
            hfb_lstm = high_flow_bias(true_hbv_flow['streamflow'][val_pd:], real_lstm_flow['streamflow'][val_pd:])

            #for precipitation
            nse_precip = nse(true_precip['PRECIP'][val_pd:], real_precip['PRECIP'][val_pd:])
            rmse_precip = rmse(true_precip['PRECIP'][val_pd:], real_precip['PRECIP'][val_pd:])
            pbias_precip = pbias(true_precip['PRECIP'][val_pd:], real_precip['PRECIP'][val_pd:])
            kge_precip = kge(true_precip['PRECIP'][val_pd:], real_precip['PRECIP'][val_pd:])

            #save the results in a dataframe
            df_result = pd.DataFrame({'station_id':[id], 'grid':[grid], 'combination':[combination],
                                        'NSE(HBV)':[nse_hbv], 'RMSE(HBV)':[rmse_hbv], 'BIAS(HBV)':[pbias_hbv], 'KGE(HBV)':[kge_hbv], 'HFB(HBV)':[hfb_hbv],
                                        'NSE(RECAL_HBV)':[nse_recal_hbv], 'RMSE(RECAL_HBV)':[rmse_recal_hbv], 'BIAS(RECAL_HBV)':[pbias_recal_hbv], 'KGE(RECAL_HBV)':[kge_recal_hbv], 'HFB(RECAL_HBV)':[hfb_recal_hbv],
                                        'NSE(HYMOD)':[nse_hymod], 'RMSE(HYMOD)':[rmse_hymod], 'BIAS(HYMOD)':[pbias_hymod], 'KGE(HYMOD)':[kge_hymod], 'HFB(HYMOD)':[hfb_hymod],
                                        'NSE(LSTM)':[nse_lstm], 'RMSE(LSTM)':[rmse_lstm], 'BIAS(LSTM)':[pbias_lstm], 'KGE(LSTM)':[kge_lstm], 'HFB(LSTM)':[hfb_lstm],
                                        'NSE(PRECIP)':[nse_precip], 'RMSE(PRECIP)':[rmse_precip], 'BIAS(PRECIP)':[pbias_precip], 'KGE(PRECIP)':[kge_precip]})
            
            df_total = pd.concat([df_total, df_result], axis=0)
    #End of loop 23
    
df_total.to_csv('output/diagnostics_validperiod.csv', index=False)    
#End of loop 1
