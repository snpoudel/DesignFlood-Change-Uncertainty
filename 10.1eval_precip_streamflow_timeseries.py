import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#read the list of basin ID
id = '01108000'

#########--TEST FOR PRECIPTATION--#########
#read true precip
true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')
true_precip['time'] = pd.to_datetime(true_precip['DATE'])
#read corresponding interpolated precip
interpol_precip2 = pd.read_csv(f'data/idw_precip/idw_precip{id}_coverage1_comb0.csv')
interpol_precip8 = pd.read_csv(f'data/idw_precip/idw_precip{id}_coverage10_comb0.csv')

#plot window
plot_start = 2165 #start at least from second year to match with LSTM streamflow
plot_end = 2515

#plot the true and interpolated precip
fig, axes = plt.subplots(4,1, figsize = (8,8), sharex=True)
axes[0].plot(true_precip['time'][plot_start:plot_end], true_precip['PRECIP'][plot_start:plot_end])
axes[0].plot(true_precip['time'][plot_start:plot_end], interpol_precip2['PRECIP'][plot_start:plot_end], linestyle = ':')
axes[0].plot(true_precip['time'][plot_start:plot_end], interpol_precip8['PRECIP'][plot_start:plot_end], linestyle = '--')
#axes[0].legend(['true precip', 'interpolated precip (1 gauge)', 'interpolated precip (n-1 gauge)'])
#axes[0].set_xlabel("Date")
axes[0].set_ylabel("Precipitation(mm/day)")


#########--TEST FOR HBV STREAMFLOW--#########
#read true streamflow
true_streamflow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
true_streamflow['date'] = pd.to_datetime(true_streamflow['date'])
#read corresponding interpolated hbv streamflow
interpol_streamflow2 = pd.read_csv(f'output/hbv_idw_streamflow/hbv_idw_streamflow{id}_coverage1_comb0.csv')
interpol_streamflow8 = pd.read_csv(f'output/hbv_idw_streamflow/hbv_idw_streamflow{id}_coverage10_comb0.csv')

#plot the true and interpolated streamflow
axes[1].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[1].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
#axes[1].legend(['true streamflow', 'interpolated streamflow (0.05 grids)', 'interpolated streamflow (0.8 grids)'])
#axes[1].set_xlabel("Date")
axes[1].set_ylabel("HBV flow(mm/day)")

#########--TEST FOR HYMOD STREAMFLOW--#########
#read corresponding interpolated hymod streamflow
interpol_streamflow2 = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{id}_coverage1_comb0.csv')
interpol_streamflow8 = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{id}_coverage10_comb0.csv')

#plot the true and interpolated streamflow
axes[2].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start:plot_end], linestyle = ':')
axes[2].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start:plot_end], linestyle = '--')
#axes[2].legend(['true streamflow', 'interpolated streamflow (0.05 grids)', 'interpolated streamflow (0.8 grids)'])
#axes[2].set_xlabel("Date")
axes[2].set_ylabel("HYMOD flow(mm/day)")

#########--TEST FOR LSTM STREAMFLOW--#########
#read corresponding interpolated lstm streamflow
interpol_streamflow2 = pd.read_csv(f'output/lstm_idw_streamflow/lstm_idw_{id}_coverage1_comb0.csv')
interpol_streamflow8 = pd.read_csv(f'output/lstm_idw_streamflow/lstm_idw_{id}_coverage10_comb0.csv')

#plot the true and interpolated streamflow
axes[3].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end])
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow2['streamflow'][plot_start-365:plot_end-365], linestyle = ':')
axes[3].plot(true_streamflow['date'][plot_start:plot_end], interpol_streamflow8['streamflow'][plot_start-365:plot_end-365], linestyle = '--')
axes[3].legend(['true', f'interpolated with 1 gauge', f'interpolated with (total-1) gauge'],
               loc='upper left', bbox_to_anchor=(0.35, -0.3))
axes[3].set_xlabel("Date")
axes[3].set_ylabel("LSTM flow(mm/day)")
plt.tight_layout()
plt.show()

#save the plot
fig.savefig('output/precip_streamflow_timeseries.png', dpi = 300)