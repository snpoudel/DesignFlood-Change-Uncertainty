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

def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value

###---step 02---###
basin_list = pd.read_csv("data/ma29basins.csv", dtype={'basin_id':str})
used_basin_list = basin_list['basin_id']

df_total = pd.DataFrame() #create an empty dataframe to store the results
for id in used_basin_list:
    # id = '01109060'
    grid_coverage = np.arange(12)
    grid_coverage = np.append(grid_coverage, [99])

    #Loop through each basin
    for grid in grid_coverage: #1
        #read true streamflow
        true_gr4j_flow = pd.read_csv(f'output/baseline/gr4j_true/gr4j_true{id}.csv')
        #keep between date 26-1-1 to 40-12-31
        start_date = true_gr4j_flow[true_gr4j_flow['date'] == '26-1-1'].index[0]
        end_date = true_gr4j_flow[true_gr4j_flow['date'] == '40-12-31'].index[0]
        true_gr4j_flow = true_gr4j_flow[start_date:end_date+1].reset_index(drop=True)

        #read true precipitation
        true_precip = pd.read_csv(f'data/baseline/true_precip/true_precip{id}.csv')

        for combination in range(10): #2
            #Read real streamflow from interpolated precipitation
            file_path = f'output/baseline/regr4j/regr4j{id}_coverage{grid}_comb{combination}.csv'
            if os.path.exists(file_path):

                #read recalibrated gr4j flow
                recal_gr4j_flow = pd.read_csv(f'output/baseline/regr4j/regr4j{id}_coverage{grid}_comb{combination}.csv')
                recal_gr4j_flow = recal_gr4j_flow[start_date:end_date+1].reset_index(drop=True)

                #read simplified hymod flow
                simp_hymod_flow = pd.read_csv(f'output/baseline/simp_hymod/simp_hymod{id}_coverage{grid}_comb{combination}.csv')
                simp_hymod_flow = simp_hymod_flow[start_date:end_date+1].reset_index(drop=True)

                #read real lstm flow for different cases
                if os.path.exists(f'output/baseline/regional_lstm/lstm_input{id}_coverage{grid}_comb{combination}.csv'):
                    real_lstm_flow = pd.read_csv(f'output/baseline/regional_lstm/lstm_input{id}_coverage{grid}_comb{combination}.csv')
                    start_date_lstm = real_lstm_flow[real_lstm_flow['date'] == '26-1-1'].index[0]
                    end_date_lstm = real_lstm_flow[real_lstm_flow['date'] == '40-12-31'].index[0]
                    real_lstm_flow = real_lstm_flow[start_date_lstm:end_date_lstm+1].reset_index(drop=True)
                    
                    real_lstm_simp_hymod_flow = pd.read_csv(f'output/baseline/regional_lstm_simp_hymod/lstm_input{id}_coverage{grid}_comb{combination}.csv')
                    real_lstm_simp_hymod_flow = real_lstm_simp_hymod_flow[start_date_lstm:end_date_lstm+1].reset_index(drop=True)

                #read real precipitation
                real_precip = pd.read_csv(f'data/baseline/noisy_precip/noisy_precip{id}_coverage{grid}_comb{combination}.csv')

                #now calculate nse, rmse, pbias, kge, hfb for real gr4j and hymod streamflow against true gr4j streamflow
                #calculate results only for test period
                #for recalibrated gr4j model
                nse_recal_gr4j = nse(true_gr4j_flow['streamflow'], recal_gr4j_flow['streamflow'])
                #for simplified hymod model
                nse_sim_hymod = nse(true_gr4j_flow['streamflow'], simp_hymod_flow['streamflow'])
                
                #for lstm model
                if os.path.exists(f'output/baseline/regional_lstm/lstm_input{id}_coverage{grid}_comb{combination}.csv'):
                    nse_lstm = nse(true_gr4j_flow['streamflow'], real_lstm_flow['streamflow'])
                else:
                    nse_lstm = np.NAN

                #for lstm-simpler-hymod model
                if os.path.exists(f'output/baseline/regional_lstm_simp_hymod/lstm_input{id}_coverage{grid}_comb{combination}.csv'):
                    nse_lstm_simp_hymod = nse(true_gr4j_flow['streamflow'], real_lstm_simp_hymod_flow['sim_streamflow'])
                else:
                    nse_lstm_simp_hymod = np.NAN

                #for precipitation
                rmse_precip = rmse(true_precip['PRECIP'], real_precip['PRECIP'])
                if np.array_equal(true_precip['PRECIP'], real_precip['PRECIP']):
                    rmse_precip = 0

                #save the results in a dataframe
                df_result = pd.DataFrame({'station_id':[id], 'grid':[grid], 'combination':[combination],
                                            'NSE(Re-gr4j)':[nse_recal_gr4j], 
                                            'NSE(Hymod)':[nse_sim_hymod], 
                                            'NSE(LSTM)':[nse_lstm], 
                                            'NSE(Hymod-LSTM)':[nse_lstm_simp_hymod], 
                                            'RMSE(PRECIP)':[rmse_precip]})
                
                df_total = pd.concat([df_total, df_result], axis=0)
        #End of loop 23
        
df_total.to_csv(f'output/eval_metrics.csv', index=False)    
#End of loop 1


##make nse plot by precip rmse categories
#read input
df_zeroprecip = df_total[df_total['RMSE(PRECIP)'] == 0] #filter precip zero
df_zeroprecip['precip_cat'] = '0'

df = df_total[df_total['RMSE(PRECIP)'] != 0] #filter everything except precip zero
#convert precipitation error into categorical group
df['precip_cat']  = pd.cut(df['RMSE(PRECIP)'], bins=[0,1,2,3,4,6,8],
                           labels=['0-1', '1-2', '2-3', '3-4', '4-6', '6-8'])

#merge back zero precips
df = pd.concat([df,df_zeroprecip], ignore_index=True)
# model_order = ['NSE(Re-gr4j)', 'NSE(LSTM)', 'NSE(Hymod)', 'NSE(Hymod-LSTM)']

#Change model name for better visualization
df = df.rename(columns={'NSE(Re-gr4j)':'GR4J-Recalib',
                        'NSE(LSTM)':'LSTM',
                        'NSE(Hymod-LSTM)':'Hymod(PP)',
                        'NSE(Hymod)':'HYMOD'})
model_order = ['GR4J-Recalib', 'LSTM', 'Hymod(PP)', 'HYMOD']
precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("colorblind")

plt.figure(figsize=(6, 4))
# Melt the dataframe for seaborn boxplot
df_melt = df.melt(id_vars=['precip_cat'], value_vars=model_order, var_name='Model', value_name='NSE')
df_melt['precip_cat'] = pd.Categorical(df_melt['precip_cat'], categories=precip_cat_order, ordered=True)
df_melt['Model'] = pd.Categorical(df_melt['Model'], categories=model_order, ordered=True)

sns.boxplot(data=df_melt, x='precip_cat', y='NSE', hue='Model', showfliers=False, width=0.8)
# sns.stripplot(data=df_melt, x='precip_cat', y='NSE', hue='Model', dodge=True, color='black', alpha=0.5, size=3)

plt.xlabel('Precipitation Error (RMSE, mm/day)')
plt.ylabel('Nash-Sutcliffe Efficiency (NSE)')
plt.ylim(0, None)
# plt.title('NSE by Precipitation RMSE Categories and Model')
plt.legend(title='Model', loc='lower left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('output/figures/nse_eval.jpg', dpi=300)
plt.show()
