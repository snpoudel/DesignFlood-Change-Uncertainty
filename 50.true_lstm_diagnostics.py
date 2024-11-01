import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv', dtype={'basin_id':str})['basin_id']
# basin_list = ['01175500', '01177000', '01181000', '01183500', '01197000', '01197500', '01198125', '01331500', '01332500']
def nse(q_obs, q_sim):
    numerator = np.sum((q_obs - q_sim)**2)
    denominator = np.sum((q_obs - (np.mean(q_obs)))**2)
    nse_value = 1 - (numerator/denominator)
    return nse_value

val_pd = 5115
val_pd_future = 365
df = pd.DataFrame()

for id in basin_list:
    #read true flow
    #historical
    # true_hbv_flow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
    # true_hbv_flow = true_hbv_flow[365:] #remove the first 365 days
    # true_hbv_flow = true_hbv_flow.reset_index(drop=True)
    

    lstm_flow = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage99_comb0.csv')
    #divide into train and test
    lstm_flow_train = lstm_flow[:val_pd]
    lstm_flow_valid = lstm_flow[val_pd:]

    nse_lstm_hist_train = nse(lstm_flow_train['true_streamflow'], lstm_flow_train['streamflow'])
    nse_lstm_hist_valid = nse(lstm_flow_valid['true_streamflow'], lstm_flow_valid['streamflow'])

    #future
    # future_hbv_flow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
    # future_hbv_flow = future_hbv_flow[365:] #remove the first 365 days
    # future_hbv_flow = future_hbv_flow.reset_index(drop=True)

    future_lstm_flow = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}_coverage99_comb0.csv')

    nse_lstm_future = nse(future_lstm_flow['true_streamflow'], future_lstm_flow['streamflow'])

    #save into a dataframe
    temp_df = pd.DataFrame({'station':id, 'nse_hist_train':nse_lstm_hist_train, 'nse_hist_valid':nse_lstm_hist_valid, 'nse_future':nse_lstm_future}, index=[0])
    df = pd.concat([df,temp_df], ignore_index=True)

# df_sort = df.sort_values('nse_hist_valid', ascending=False).reset_index(drop=True)
# print(df_sort.head(29))


#NSE
sns.kdeplot(data = df['nse_hist_train'], cumulative=True, bw_method = 0.01, label='Train')
sns.kdeplot(data = df['nse_hist_valid'], cumulative=True, bw_method = 0.01, label='Test')
sns.kdeplot(data = df['nse_future'], cumulative=True,bw_method = 0.01, label='NSE Future')
plt.legend()
plt.xlabel('NSE')
plt.ylabel('CDF')
plt.grid(True)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

#read lstm parameters from sungwook
# params_sungwook = pd.read_csv('sungwook_lstm.csv')
# sns.kdeplot(data = params_sungwook['NSE_Train'], cumulative=True, bw_method = 0.01, label='Train Sungwook', linestyle='--')
# sns.kdeplot(data = params_sungwook['NSE_Test'], cumulative=True, bw_method = 0.01, label='Test Sungwook')