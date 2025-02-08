import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#read the list of basin ID
id = '01108000'

#write a function to calculate RMSE
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value

#choose coverage and combination
cov1 = 2
cov2 = 8
combination = 8

####--HISTORICAL--####
#########--TEST FOR PRECIPTATION--#########
#read true precip
true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')
# true_precip['time'] = pd.to_datetime(true_precip['DATE'])
#read corresponding interpolated precip
interpol_precip2 = pd.read_csv(f'data/noisy_precip/noisy_precip{id}_coverage{cov1}_comb{combination}.csv')
#find the RMSE
precip_rmse2 = rmse(true_precip['PRECIP'], interpol_precip2['PRECIP'])
precip_rmse2 = round(precip_rmse2, 1)
interpol_precip8 = pd.read_csv(f'data/noisy_precip/noisy_precip{id}_coverage{cov2}_comb{combination}.csv')
#find the RMSE
precip_rmse8 = rmse(true_precip['PRECIP'], interpol_precip8['PRECIP'])
precip_rmse8 = round(precip_rmse8, 1)

# print(precip_rmse2, precip_rmse8)
#plot window
plot_start = 10030 #start at least from second year to match with LSTM streamflow
plot_end = plot_start + 60

#plot the true and interpolated precip
fig, axes = plt.subplots(7,1, figsize = (12,12), sharex=True)
plt.suptitle(f'Historical data,   Basin ID: {id}')
axes[0].plot(true_precip['DATE'][plot_start:plot_end], true_precip['PRECIP'][plot_start:plot_end])
axes[0].plot(true_precip['DATE'][plot_start:plot_end], interpol_precip2['PRECIP'][plot_start:plot_end], linestyle = ':')
axes[0].plot(true_precip['DATE'][plot_start:plot_end], interpol_precip8['PRECIP'][plot_start:plot_end], linestyle = '--')
axes[0].legend(['Truth, 10gauges',
             f'Noisy, gauges={cov1}, error ={precip_rmse2}',
             f'Noisy, gauges={cov2}, error ={precip_rmse8}'],
             loc='best', )
axes[0].set_ylabel("Precipitation")
axes[0].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR Re-HBV STREAMFLOW--#########
#read true streamflow
true_streamflow = pd.read_csv(f'output/hbv_true/hbv_true{id}.csv')
# true_streamflow['date'] = pd.to_datetime(true_streamflow['date'])
#read corresponding interpolated hbv streamflow
interpol_streamflow2 = pd.read_csv(f'output/rehbv/rehbv{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/rehbv/rehbv{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[1].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
axes[1].set_ylabel("Re-HBV flow")
axes[1].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR HYMOD STREAMFLOW--#########
#read corresponding interpolated hymod streamflow
interpol_streamflow2 = pd.read_csv(f'output/hymod/hymod{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/hymod/hymod{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[2].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
#axes[2].legend(['true streamflow', 'interpolated streamflow (0.05 grids)', 'interpolated streamflow (0.8 grids)'])
#axes[2].set_xlabel("Date")
axes[2].set_ylabel("HYMOD flow")
axes[2].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR LSTM STREAMFLOW--#########
#read corresponding interpolated lstm streamflow
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[3].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start-364:plot_end-364], linestyle = '--')
# axes[3].set_xlabel("Date")
axes[3].set_ylabel("LSTM flow")
axes[3].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR HYMOD LSTM STREAMFLOW--#########
#read corresponding interpolated simphyd streamflow
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm_hymod/final_output/historical/hymod_lstm{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm_hymod/final_output/historical/hymod_lstm{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[4].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[4].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[4].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = '--')
# axes[4].set_xlabel("Date")
axes[4].set_ylabel("HYMOD-LSTM flow")
axes[4].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR SIMP HYMOD LSTM STREAMFLOW--#########
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm_simp_hymod/final_output/historical/hymod_lstm{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm_simp_hymod/final_output/historical/hymod_lstm{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[5].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[5].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['simp_hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[5].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['simp_hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = '--')
axes[5].set_ylabel("SIMP-HYMOD-LSTM flow")
axes[5].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR SIMP HYMOD STREAMFLOW--#########
interpol_streamflow2 = pd.read_csv(f'output/simp_hymod/simp_hymod{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/simp_hymod/simp_hymod{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[6].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[6].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[6].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
axes[6].set_ylabel("SIMP-HYMOD flow")
#set x ticks as a range of values from 1 to 90
# axes[6].set_xticks(range(1,91))
axes[6].set_xlabel("Date")
axes[6].grid(True, linestyle='--', alpha = 0.5)
axes[6].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()

#save the plot
fig.savefig(f'output/figures/HistTimeseries{id}.png', dpi = 300)



####--FUTURE--####
#########--TEST FOR PRECIPTATION--#########
#read true precip
true_precip = pd.read_csv(f'data/future/future_true_precip/future_true_precip{id}.csv')
# true_precip['time'] = pd.to_datetime(true_precip['DATE'])
#read corresponding interpolated precip
interpol_precip2 = pd.read_csv(f'data/future/future_noisy_precip/future_noisy_precip{id}_coverage{cov1}_comb{combination}.csv')
interpol_precip8 = pd.read_csv(f'data/future/future_noisy_precip/future_noisy_precip{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated precip
fig, axes = plt.subplots(7,1, figsize = (12,12), sharex=True)
plt.suptitle(f'Future data,   Basin ID: {id}')
axes[0].plot(true_precip['DATE'][plot_start:plot_end], true_precip['PRECIP'][plot_start:plot_end])
axes[0].plot(true_precip['DATE'][plot_start:plot_end], interpol_precip2['PRECIP'][plot_start:plot_end], linestyle = ':')
axes[0].plot(true_precip['DATE'][plot_start:plot_end], interpol_precip8['PRECIP'][plot_start:plot_end], linestyle = '--')
axes[0].legend(['Truth, 10gauges',
             f'Noisy, gauges={cov1}, error ={precip_rmse2}',
             f'Noisy, gauges={cov2}, error ={precip_rmse8}'],
             loc='best', )
axes[0].set_ylabel("Precipitation")
axes[0].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR Re-HBV STREAMFLOW--#########
#read true streamflow
true_streamflow = pd.read_csv(f'output/future/hbv_true/hbv_true{id}.csv')
# true_streamflow['date'] = pd.to_datetime(true_streamflow['date'])
#read corresponding interpolated hbv streamflow
interpol_streamflow2 = pd.read_csv(f'output/future/rehbv/rehbv{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/future/rehbv/rehbv{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[1].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
axes[1].set_ylabel("Re-HBV flow")
axes[1].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR HYMOD STREAMFLOW--#########
#read corresponding interpolated hymod streamflow
interpol_streamflow2 = pd.read_csv(f'output/future/hymod/hymod{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/future/hymod/hymod{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[2].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
#axes[2].legend(['true streamflow', 'interpolated streamflow (0.05 grids)', 'interpolated streamflow (0.8 grids)'])
#axes[2].set_xlabel("Date")
axes[2].set_ylabel("HYMOD flow")
axes[2].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR LSTM STREAMFLOW--#########
#read corresponding interpolated lstm streamflow
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[3].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start-364:plot_end-364], linestyle = '--')
# axes[3].set_xlabel("Date")
axes[3].set_ylabel("LSTM flow")
axes[3].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR HYMOD LSTM STREAMFLOW--#########
#read corresponding interpolated simphyd streamflow
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm_hymod/final_output/future/hymod_lstm{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm_hymod/final_output/future/hymod_lstm{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[4].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[4].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[4].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = '--')
# axes[4].set_xlabel("Date")
axes[4].set_ylabel("HYMOD-LSTM flow")
axes[4].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR SIMP HYMOD LSTM STREAMFLOW--#########
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm_simp_hymod/final_output/future/hymod_lstm{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm_simp_hymod/final_output/future/hymod_lstm{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[5].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[5].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['simp_hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[5].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['simp_hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = '--')
axes[5].set_ylabel("SIMP-HYMOD-LSTM flow")
axes[5].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR SIMP HYMOD STREAMFLOW--#########
interpol_streamflow2 = pd.read_csv(f'output/future/simp_hymod/simp_hymod{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/future/simp_hymod/simp_hymod{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[6].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[6].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[6].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
axes[6].set_ylabel("SIMP-HYMOD flow")
#set x ticks as a range of values from 1 to 90
# axes[6].set_xticks(range(1,91))
axes[6].set_xlabel("Date")
axes[6].grid(True, linestyle='--', alpha = 0.5)
axes[6].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()

#save the plot
fig.savefig(f'output/figures/FutTimeseries{id}.png', dpi = 300)


