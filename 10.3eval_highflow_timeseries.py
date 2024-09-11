import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#read data
basin_id = '01108000'
grid = 10
comb = 1

#precip
true_precip = pd.read_csv(f'data/true_precip/true_precip{basin_id}.csv')
idw_precip = pd.read_csv(f'data/idw_precip/idw_precip{basin_id}_coverage{grid}_comb{comb}.csv')

future_true_precip = pd.read_csv(f'data/future/future_true_precip/future_true_precip{basin_id}.csv')
future_idw_precip = pd.read_csv(f'data/future/future_idw_precip/future_idw_precip{basin_id}_coverage{grid}_comb{comb}.csv')

precip_rmse = np.sqrt(np.mean((true_precip['PRECIP'] - idw_precip['PRECIP'])**2)) #calculate the rmse
precip_rmse = np.round(precip_rmse, 2)


#true flow
hbv_truth = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{basin_id}.csv')
hbv_truth['date'] = pd.to_datetime(hbv_truth['date']).dt.year
dhbv_truth = hbv_truth.groupby('date')['streamflow'].agg('max')

future_hbv_truth = pd.read_csv(f'output/future/hbv_true_future_streamflow/hbv_true_future_output_{basin_id}.csv')
future_hbv_truth['date'] = pd.to_datetime(future_hbv_truth['date']).dt.year
future_dhbv_truth = future_hbv_truth.groupby('date')['streamflow'].agg('max')



#hymod flow
hymod = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{basin_id}_coverage{grid}_comb{comb}.csv')
hymod['date'] = pd.to_datetime(hymod['date']).dt.year
dhymod = hymod.groupby('date')['streamflow'].agg('max')

future_hymod = pd.read_csv(f'output/future/hymod_idw_future_streamflow/hymod_interpol_future_streamflow{basin_id}_coverage{grid}_comb{comb}.csv')
future_hymod['date'] = pd.to_datetime(future_hymod['date']).dt.year
future_dhymod = future_hymod.groupby('date')['streamflow'].agg('max')

#plot
fig, axs = plt.subplots(figsize=(10,6))
plt.plot(dhbv_truth, label='Truth Hist', color='blue')
plt.plot(future_dhbv_truth, label='Truth Future', color ='orange')
plt.plot(dhymod, label='Hymod Hist', color='blue', linestyle='--')
plt.plot(future_dhymod, label='Hymod Future', color='orange', linestyle='--')
plt.legend()
plt.ylabel('Annual maximum streamflow')
plt.xlabel('Year')
plt.title(f'Precip RMSE: {precip_rmse}')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


#save the plot
fig.savefig('output/eval_highflow_timeseries.png', dpi = 300)
