import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv', dtype={'basin_id':str})

def nse(q_obs, q_sim):
    numerator = np.sum((q_obs - q_sim)**2)
    denominator = np.sum((q_obs - (np.mean(q_obs)))**2)
    nse_value = 1 - (numerator/denominator)
    return nse_value

val_pd = 5115
val_pd_future = 365
df = pd.DataFrame()
for id in basin_list['basin_id']:
    #read true flow
    #historical
    true_hbv_flow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
    true_hbv_flow = true_hbv_flow[365:] #remove the first 365 days
    true_hbv_flow = true_hbv_flow.reset_index(drop=True)

    lstm_flow = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}.csv')

    nse_lstm_hist_train = nse(true_hbv_flow['streamflow'][0:val_pd], lstm_flow['streamflow'][0:val_pd])
    nse_lstm_hist_valid = nse(true_hbv_flow['streamflow'][val_pd:], lstm_flow['streamflow'][val_pd:])

    #future
    future_hbv_flow = pd.read_csv(f'Z:/MA-Precip-Uncertainty-GaugeData/output/hbv_true_streamflow/hbv_true_output_{id}.csv')
    future_hbv_flow = future_hbv_flow[365:] #remove the first 365 days
    future_hbv_flow = future_hbv_flow.reset_index(drop=True)

    future_lstm_flow = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}.csv')

    nse_lstm_future = nse(future_hbv_flow['streamflow'][val_pd_future:], future_lstm_flow['streamflow'][val_pd_future:])

    #save into a dataframe
    temp_df = pd.DataFrame({'station':id, 'nse_hist_train':nse_lstm_hist_train, 'nse_hist_valid':nse_lstm_hist_valid, 'nse_future':nse_lstm_future}, index=[0])
    df = pd.concat([df,temp_df], ignore_index=True)
#df.to_csv('output/true_lstm_diagnostics.csv', index=False)
# df = df.dropna()
df_sort = df.sort_values('nse_hist_train', ascending=False).reset_index(drop=True)
print(df_sort)

#Make CDF plot of NSE for calibration period
#NSE
sns.kdeplot(data = df['nse_hist_train'], cumulative=True, bw_method = 0.05, label='NSE Hist Train')
sns.kdeplot(data = df['nse_hist_valid'], cumulative=True, bw_method = 0.05, label='NSE Hist Valid')
sns.kdeplot(data = df['nse_future'], cumulative=True,bw_method = 0.05, label='NSE Future')
plt.legend()
plt.show()