import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#read the list of basin ID
id = '01108000'

#write a function to calculate RMSE
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value

#choose coverage and combination
cov1 = 2
cov2 = 8
combination = 2
####--HISTORICAL--####
#########--TEST FOR PRECIPTATION--#########
#read true precip
true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')
true_precip['time'] = pd.to_datetime(true_precip['DATE'])
#read corresponding interpolated precip
interpol_precip2 = pd.read_csv(f'data/idw_precip/idw_precip{id}_coverage{cov1}_comb{combination}.csv')
#find the RMSE
precip_rmse2 = rmse(true_precip['PRECIP'], interpol_precip2['PRECIP'])
precip_rmse2 = round(precip_rmse2, 2)
interpol_precip8 = pd.read_csv(f'data/idw_precip/idw_precip{id}_coverage{cov2}_comb{combination}.csv')
#find the RMSE
precip_rmse8 = rmse(true_precip['PRECIP'], interpol_precip8['PRECIP'])
precip_rmse8 = round(precip_rmse8, 2)

#plot window
plot_start = 3730 #start at least from second year to match with LSTM streamflow
plot_end = 3760

#plot the true and interpolated precip
fig, axes = plt.subplots(4,1, figsize = (10,8), sharex=True)
plt.suptitle(f'Historical data,   Basin ID: {id}')
axes[0].plot(true_precip['time'][plot_start:plot_end], true_precip['PRECIP'][plot_start:plot_end])
axes[0].plot(true_precip['time'][plot_start:plot_end], interpol_precip2['PRECIP'][plot_start:plot_end], linestyle = ':')
axes[0].plot(true_precip['time'][plot_start:plot_end], interpol_precip8['PRECIP'][plot_start:plot_end], linestyle = '--')
axes[0].legend(['Truth, 10gauges',
             f'interpolated, gauges={cov1}, error ={precip_rmse2}',
             f'interpolated, gauges={cov2}, error ={precip_rmse8}'],
             loc='best', )
axes[0].set_ylabel("Precipitation(mm/day)")
axes[0].grid(True)


#########--TEST FOR Re-HBV STREAMFLOW--#########
#read true streamflow
true_streamflow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
true_streamflow['date'] = pd.to_datetime(true_streamflow['date'])
#read corresponding interpolated hbv streamflow
interpol_streamflow2 = pd.read_csv(f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[1].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
axes[1].set_ylabel("Re-HBV flow(mm/day)")
axes[1].grid(True)
#########--TEST FOR HYMOD STREAMFLOW--#########
#read corresponding interpolated hymod streamflow
interpol_streamflow2 = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[2].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
#axes[2].legend(['true streamflow', 'interpolated streamflow (0.05 grids)', 'interpolated streamflow (0.8 grids)'])
#axes[2].set_xlabel("Date")
axes[2].set_ylabel("HYMOD flow(mm/day)")
axes[2].grid(True)
#########--TEST FOR LSTM STREAMFLOW--#########
#read corresponding interpolated lstm streamflow
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[3].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start-364:plot_end-364], linestyle = '--')
axes[3].set_xlabel("Date")
axes[3].set_ylabel("LSTM flow(mm/day)")
axes[3].grid(True)
plt.tight_layout()
plt.show()

#save the plot
fig.savefig(f'output/figures/{id}/HistTimeseries_precip_streamflow.png', dpi = 300)





####--FUTURE--####
#########--TEST FOR PRECIPTATION--#########
#read true precip
true_precip = pd.read_csv(f'data/future/future_true_precip/future_true_precip{id}.csv')
true_precip['time'] = pd.to_datetime(true_precip['DATE'])
#read corresponding interpolated precip
interpol_precip2 = pd.read_csv(f'data/future/future_idw_precip/future_idw_precip{id}_coverage{cov1}_comb{combination}.csv')
#find the RMSE
# precip_rmse2 = rmse(true_precip['PRECIP'], interpol_precip2['PRECIP'])
# precip_rmse2 = round(precip_rmse2, 2)
interpol_precip8 = pd.read_csv(f'data/future/future_idw_precip/future_idw_precip{id}_coverage{cov2}_comb{combination}.csv')
#find the RMSE
# precip_rmse8 = rmse(true_precip['PRECIP'], interpol_precip8['PRECIP'])
# precip_rmse8 = round(precip_rmse8, 2)

#plot window
plot_start = 3730 #start at least from second year to match with LSTM streamflow
plot_end = 3760

#plot the true and interpolated precip
fig, axes = plt.subplots(4,1, figsize = (10,8), sharex=True)
plt.suptitle(f'Climate change data,   Basin ID: {id}')
axes[0].plot(true_precip['time'][plot_start:plot_end], true_precip['PRECIP'][plot_start:plot_end])
axes[0].plot(true_precip['time'][plot_start:plot_end], interpol_precip2['PRECIP'][plot_start:plot_end], linestyle = ':')
axes[0].plot(true_precip['time'][plot_start:plot_end], interpol_precip8['PRECIP'][plot_start:plot_end], linestyle = '--')
axes[0].legend(['Truth, gauges=10',
             f'interpolated, gauges={cov1}, error ={precip_rmse2}',
             f'interpolated, gauges={cov2}, error ={precip_rmse8}'],
             loc='best', )
axes[0].set_ylabel("Precipitation(mm/day)")
axes[0].grid(True)


#########--TEST FOR Re-HBV STREAMFLOW--#########
#read true streamflow
true_streamflow = pd.read_csv(f'output/future/hbv_true_future_streamflow/hbv_true_future_output_{id}.csv')
true_streamflow['date'] = pd.to_datetime(true_streamflow['date'])
#read corresponding interpolated hbv streamflow
interpol_streamflow2 = pd.read_csv(f'output/future/hbv_idw_recalib_future_streamflow/hbv_idw_recalib_future_streamflow{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/future/hbv_idw_recalib_future_streamflow/hbv_idw_recalib_future_streamflow{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[1].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
axes[1].set_ylabel("Re-HBV flow(mm/day)")
axes[1].grid(True)
#########--TEST FOR HYMOD STREAMFLOW--#########
#read corresponding interpolated hymod streamflow
interpol_streamflow2 = pd.read_csv(f'output/future/hymod_idw_future_streamflow/hymod_interpol_future_streamflow{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/future/hymod_idw_future_streamflow/hymod_interpol_future_streamflow{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[2].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
#axes[2].legend(['true streamflow', 'interpolated streamflow (0.05 grids)', 'interpolated streamflow (0.8 grids)'])
#axes[2].set_xlabel("Date")
axes[2].set_ylabel("HYMOD flow(mm/day)")
axes[2].grid(True)
#########--TEST FOR LSTM STREAMFLOW--#########
#read corresponding interpolated lstm streamflow
interpol_streamflow2 = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}_coverage{cov1}_comb{combination}.csv')
interpol_streamflow8 = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}_coverage{cov2}_comb{combination}.csv')

#plot the true and interpolated streamflow
axes[3].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start-364:plot_end-364], linestyle = ':')
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start-364:plot_end-364], linestyle = '--')
axes[3].set_xlabel("Date")
axes[3].set_ylabel("LSTM flow(mm/day)")
axes[3].grid(True)
plt.tight_layout()
plt.show()

#save the plot
fig.savefig(f'output/figures/{id}/FutureTimeseries_precip_streamflow.png', dpi = 300)