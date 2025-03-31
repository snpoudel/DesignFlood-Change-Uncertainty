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
    q_obs_995_value = np.percentile(q_obs, 99.9)
    indices_q995 = np.where(q_obs > q_obs_995_value)
    q_obs_995 = q_obs[indices_q995]
    q_sim_995 = q_sim[indices_q995]
    hfb = (np.sum(q_obs_995 - q_sim_995) / np.sum(q_obs_995)) * 100
    return hfb


###---step 02---###
basin_list = pd.read_csv("Z:/MA-Precip-Uncertainty-GaugeData/data/ma29basins.csv", dtype={'basin_id':str})
used_basin_list = basin_list['basin_id']

df_total = pd.DataFrame() #create an empty dataframe to store the results
for id in used_basin_list:
    # id = '01109060'
    grid_coverage = np.arange(12)
    grid_coverage = np.append(grid_coverage, [99])

    #Loop through each basin
    for grid in grid_coverage: #1
        #read true streamflow
        true_hbv_flow = pd.read_csv(f'output/hbv_true/hbv_true{id}.csv')
        #keep between date 26-1-1 to 40-12-31
        start_date = true_hbv_flow[true_hbv_flow['date'] == '26-1-1'].index[0]
        end_date = true_hbv_flow[true_hbv_flow['date'] == '40-12-31'].index[0]
        true_hbv_flow = true_hbv_flow[start_date:end_date+1].reset_index(drop=True)

        #read true precipitation
        true_precip = pd.read_csv(f'Z:/MA-Precip-Uncertainty-GaugeData/data/true_precip/true_precip{id}.csv')

        for combination in range(10): #2
            #Read real streamflow from interpolated precipitation
            file_path = f'output/rehbv/rehbv{id}_coverage{grid}_comb{combination}.csv'
            if os.path.exists(file_path):
                #read recalibrated hbv flow
                recal_hbv_flow = pd.read_csv(f'output/rehbv/rehbv{id}_coverage{grid}_comb{combination}.csv')
                recal_hbv_flow = recal_hbv_flow[start_date:end_date+1].reset_index(drop=True)

                #read real hymod flow
                real_hymod_flow = pd.read_csv(f'output/hymod/hymod{id}_coverage{grid}_comb{combination}.csv')
                real_hymod_flow = real_hymod_flow[start_date:end_date+1].reset_index(drop=True)

                #read simplified hymod flow
                simp_hymod_flow = pd.read_csv(f'output/simp_hymod/simp_hymod{id}_coverage{grid}_comb{combination}.csv')
                simp_hymod_flow = simp_hymod_flow[start_date:end_date+1].reset_index(drop=True)

                #read real precipitation
                real_precip = pd.read_csv(f'Z:/MA-Precip-Uncertainty-GaugeData/data/noisy_precip/noisy_precip{id}_coverage{grid}_comb{combination}.csv')

                #now calculate nse, rmse, pbias, kge, hfb for real hbv and hymod streamflow against true hbv streamflow
                #calculate results only for test period
                #for recalibrated hbv model
                nse_recal_hbv = nse(true_hbv_flow['streamflow'], recal_hbv_flow['streamflow'])
                rmse_recal_hbv = rmse(true_hbv_flow['streamflow'], recal_hbv_flow['streamflow'])
                pbias_recal_hbv = pbias(true_hbv_flow['streamflow'], recal_hbv_flow['streamflow'])
                kge_recal_hbv = kge(true_hbv_flow['streamflow'], recal_hbv_flow['streamflow'])
                hfb_recal_hbv = high_flow_bias(true_hbv_flow['streamflow'], recal_hbv_flow['streamflow'])
                #for hymod model
                nse_hymod = nse(true_hbv_flow['streamflow'], real_hymod_flow['streamflow'])
                rmse_hymod = rmse(true_hbv_flow['streamflow'], real_hymod_flow['streamflow'])
                pbias_hymod = pbias(true_hbv_flow['streamflow'], real_hymod_flow['streamflow'])
                kge_hymod = kge(true_hbv_flow['streamflow'], real_hymod_flow['streamflow'])
                hfb_hymod = high_flow_bias(true_hbv_flow['streamflow'], real_hymod_flow['streamflow'])
                #for simplified hymod model
                nse_sim_hymod = nse(true_hbv_flow['streamflow'], simp_hymod_flow['streamflow'])
                rmse_sim_hymod = rmse(true_hbv_flow['streamflow'], simp_hymod_flow['streamflow'])
                pbias_sim_hymod = pbias(true_hbv_flow['streamflow'], simp_hymod_flow['streamflow'])
                kge_sim_hymod = kge(true_hbv_flow['streamflow'], simp_hymod_flow['streamflow'])
                hfb_sim_hymod = high_flow_bias(true_hbv_flow['streamflow'], simp_hymod_flow['streamflow'])

                #for precipitation
                nse_precip = nse(true_precip['PRECIP'], real_precip['PRECIP'])
                rmse_precip = rmse(true_precip['PRECIP'], real_precip['PRECIP'])
                if np.array_equal(true_precip['PRECIP'], real_precip['PRECIP']):
                    rmse_precip = 0
                pbias_precip = pbias(true_precip['PRECIP'], real_precip['PRECIP'])
                kge_precip = kge(true_precip['PRECIP'], real_precip['PRECIP'])

                #save the results in a dataframe
                df_result = pd.DataFrame({'station_id':[id], 'grid':[grid], 'combination':[combination],
                                            'NSE(RECAL_HBV)':[nse_recal_hbv], 'RMSE(RECAL_HBV)':[rmse_recal_hbv], 'BIAS(RECAL_HBV)':[pbias_recal_hbv], 'KGE(RECAL_HBV)':[kge_recal_hbv], 'HFB(RECAL_HBV)':[hfb_recal_hbv],
                                            'NSE(FULL-HYMOD)':[nse_hymod], 'RMSE(FULL-HYMOD)':[rmse_hymod], 'BIAS(FULL-HYMOD)':[pbias_hymod], 'KGE(FULL-HYMOD)':[kge_hymod], 'HFB(FULL-HYMOD)':[hfb_hymod],
                                            'NSE(HYMOD)':[nse_sim_hymod], 'RMSE(HYMOD)':[rmse_sim_hymod], 'BIAS(HYMOD)':[pbias_sim_hymod], 'KGE(HYMOD)':[kge_sim_hymod], 'HFB(HYMOD)':[hfb_sim_hymod],
                                            'NSE(PRECIP)':[nse_precip], 'RMSE(PRECIP)':[rmse_precip], 'BIAS(PRECIP)':[pbias_precip], 'KGE(PRECIP)':[kge_precip]})
                
                df_total = pd.concat([df_total, df_result], axis=0)
        #End of loop 23
        
df_total.to_csv(f'output/allbasins_diagnostics_validperiod.csv', index=False)    
#End of loop 1
