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
plot_start = 10015 #start at least from second year to match with LSTM streamflow
plot_end = plot_start + 60

#plot the true and interpolated precip
fig, axes = plt.subplots(4,1, figsize = (7,7), sharex=True)
# plt.suptitle(f'Historical data,   Basin ID: {id}')
axes[0].plot(true_precip['DATE'][plot_start:plot_end], true_precip['PRECIP'][plot_start:plot_end])
axes[0].plot(true_precip['DATE'][plot_start:plot_end], interpol_precip2['PRECIP'][plot_start:plot_end], linestyle = ':')
axes[0].plot(true_precip['DATE'][plot_start:plot_end], interpol_precip8['PRECIP'][plot_start:plot_end], linestyle = '--')
axes[0].legend(['Truth, 10gauges, precip error = 0',
             f'Noisy, num. gauges={cov1}, precip error ={precip_rmse2} mm/day',
             f'Noisy, num. gauges={cov2}, precip error ={precip_rmse8} mm/day'],
             loc='best', )
axes[0].set_ylabel("Precipitation")
axes[0].grid(True, linestyle='--', alpha = 0.5)

#read true streamflow
true_streamflow = pd.read_csv(f'output/hbv_true/hbv_true{id}.csv')


#########--TEST FOR LSTM STREAMFLOW--#########
#read corresponding interpolated lstm streamflow
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage{cov2}_comb{combination}.csv')
#plot the true and interpolated streamflow
axes[3].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start-364:plot_end-364], linestyle = '--')
axes[3].set_ylabel("LSTM Flow")
axes[3].set_xlabel("Days")
axes[3].set_ylim(0,4.5)
axes[3].grid(True, linestyle='--', alpha = 0.5)


#########--TEST FOR SIMP HYMOD LSTM STREAMFLOW--#########
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm_simp_hymod/final_output/historical/hymod_lstm{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm_simp_hymod/final_output/historical/hymod_lstm{id}_coverage{cov2}_comb{combination}.csv')
#plot the true and interpolated streamflow
axes[2].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['simp_hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['simp_hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = '--')
axes[2].set_ylabel("HYMOD-LSTM Flow")
axes[2].set_ylim(0,4.5)
axes[2].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR SIMP HYMOD STREAMFLOW--#########
interpol_streamflow2 = pd.read_csv(f'output/simp_hymod/simp_hymod{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/simp_hymod/simp_hymod{id}_coverage{cov2}_comb{combination}.csv')
#plot the true and interpolated streamflow
axes[1].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
axes[1].set_ylabel("HYMOD Flow")
axes[1].set_ylim(0,4.5)
axes[1].grid(True, linestyle='--', alpha = 0.5)
axes[1].set_xticks(range(0,61,10))
#set x label values as numbers from 1 to 60
axes[1].set_xticklabels(np.array([1,11,21,31,41,51, 60]))
plt.tight_layout()
plt.show()

#save the plot
fig.savefig(f'output/figures/01108000/HistTimeseries{id}.png', dpi = 300)
#save inkcspae
fig.savefig(f'output/figures/01108000/HistTimeseries{id}.svg', dpi = 300)



####--FUTURE--####
#########--TEST FOR PRECIPTATION--#########
#read true precip
true_precip = pd.read_csv(f'data/future/future_true_precip/future_true_precip{id}.csv')
# true_precip['time'] = pd.to_datetime(true_precip['DATE'])
#read corresponding interpolated precip
interpol_precip2 = pd.read_csv(f'data/future/future_noisy_precip/future_noisy_precip{id}_coverage{cov1}_comb{combination}.csv')
interpol_precip8 = pd.read_csv(f'data/future/future_noisy_precip/future_noisy_precip{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated precip
fig, axes = plt.subplots(4,1, figsize = (7,7), sharex=True)
# plt.suptitle(f'Future data,   Basin ID: {id}')
axes[0].plot(true_precip['DATE'][plot_start:plot_end], true_precip['PRECIP'][plot_start:plot_end])
axes[0].plot(true_precip['DATE'][plot_start:plot_end], interpol_precip2['PRECIP'][plot_start:plot_end], linestyle = ':')
axes[0].plot(true_precip['DATE'][plot_start:plot_end], interpol_precip8['PRECIP'][plot_start:plot_end], linestyle = '--')
axes[0].legend(['Truth, 10gauges, precip error = 0',
             f'Noisy, num. gauges={cov1}, precip error ={precip_rmse2} mm/day',
             f'Noisy, num. gauges={cov2}, precip error ={precip_rmse8} mm/day'],
             loc='best', )
axes[0].set_ylabel("Precipitation")
axes[0].grid(True, linestyle='--', alpha = 0.5)

#read true streamflow
true_streamflow = pd.read_csv(f'output/future/hbv_true/hbv_true{id}.csv')

#########--TEST FOR LSTM STREAMFLOW--#########
#read corresponding interpolated lstm streamflow
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}_coverage{cov2}_comb{combination}.csv')
#plot the true and interpolated streamflow
axes[3].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start-364:plot_end-364], linestyle = '--')
axes[3].set_ylabel("LSTM flow")
axes[1].set_xlabel("Days")
axes[3].grid(True, linestyle='--', alpha = 0.5)


#########--TEST FOR SIMP HYMOD LSTM STREAMFLOW--#########
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm_simp_hymod/final_output/future/hymod_lstm{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm_simp_hymod/final_output/future/hymod_lstm{id}_coverage{cov2}_comb{combination}.csv')
#plot the true and interpolated streamflow
axes[2].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['simp_hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['simp_hymod_lstm_streamflow'][plot_start-364:plot_end-364], linestyle = '--')
axes[2].set_ylabel("HYMOD-LSTM flow")
axes[2].grid(True, linestyle='--', alpha = 0.5)

#########--TEST FOR SIMP HYMOD STREAMFLOW--#########
interpol_streamflow2 = pd.read_csv(f'output/future/simp_hymod/simp_hymod{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/future/simp_hymod/simp_hymod{id}_coverage{cov2}_comb{combination}.csv')
#plot the true and interpolated streamflow
axes[1].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
axes[1].set_ylabel("SIMP-HYMOD flow")
axes[1].grid(True, linestyle='--', alpha = 0.5)
axes[1].set_xticks(range(1,61,10))
plt.tight_layout()
plt.show()

#save the plot
fig.savefig(f'output/figures/01108000/FutTimeseries{id}.png', dpi = 300)


