import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#write a function to calculate percentage high flow bias
def hfb(q_obs, q_sim):
    q_obs = np.array(q_obs)
    q_sim = np.array(q_sim)
    q_obs_995 = np.percentile(q_obs, 99.9)
    indices_q995 = np.where(q_obs > q_obs_995)
    q_sim_995 = q_sim[indices_q995]
    hfb = (np.sum(q_obs_995 - q_sim_995) / np.sum(q_obs_995)) * 100
    return hfb

#read data
basin_id = '01108000'
grid = 0.4
comb = 2
true_precip = pd.read_csv(f'data/true_precip/true_precip{basin_id}.csv')
idw_precip = pd.read_csv(f'data/idw_precip/idw_precip{basin_id}_coverage{grid}_comb{comb}.csv')
precip_rmse = np.sqrt(np.mean((true_precip['PRECIP'] - idw_precip['PRECIP'])**2)) #calculate the rmse
precip_rmse = np.round(precip_rmse, 1)

hbv_truth = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{basin_id}.csv')
hbv_truth['date'] = pd.to_datetime(hbv_truth['date']).dt.year
dhbv_truth = hbv_truth.groupby('date')['streamflow'].agg('max')

hbv_real = pd.read_csv(f'output/hbv_idw_streamflow/hbv_idw_streamflow{basin_id}_coverage{grid}_comb{comb}.csv')
hbv_real['date'] = pd.to_datetime(hbv_real['date']).dt.year
dhbv_real = hbv_real.groupby('date')['streamflow'].agg('max')
hbvreal_hfb = hfb(hbv_truth['streamflow'], hbv_real['streamflow'])
hbvreal_hfb = np.round(hbvreal_hfb,1)

hbv_recalibrated = pd.read_csv(f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{basin_id}_coverage{grid}_comb{comb}.csv')
hbv_recalibrated['date'] = pd.to_datetime(hbv_recalibrated['date']).dt.year
dhbv_recalibrated = hbv_recalibrated.groupby('date')['streamflow'].agg('max')
hbvrecal_hfb = hfb(hbv_truth['streamflow'], hbv_recalibrated['streamflow'])
hbvrecal_hfb = np.round(hbvrecal_hfb,1)

hymod = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{basin_id}_coverage{grid}_comb{comb}.csv')
hymod['date'] = pd.to_datetime(hymod['date']).dt.year
dhymod = hymod.groupby('date')['streamflow'].agg('max')
hymod_hfb = hfb(hbv_truth['streamflow'], hymod['streamflow'])
hymod_hfb = np.round(hymod_hfb,1)


plt.plot(dhbv_truth, label='Truth')
plt.plot(dhbv_real, label = f'HBV')
plt.plot(dhbv_recalibrated, label=f'HBV Recal')
plt.plot(dhymod, label=f'HYMOD')
plt.legend()
plt.ylabel('Annual maximum streamflow')
plt.xlabel('Year')
plt.title(f'Precip RMSE: {precip_rmse}')
plt.grid(True, linestyle='--', alpha=0.8)
plt.show()

plt.plot(hbv_truth['streamflow'][1000:1200], label='truth')
plt.plot(hbv_real['streamflow'][1000:1200], label = f'HBV, hfb={hbvreal_hfb}')
plt.plot(hbv_recalibrated['streamflow'][1000:1200], label=f'HBV Recal, hfb={hbvrecal_hfb}')
plt.plot(hymod['streamflow'][1000:1200], label=f'HYMOD, hfb={hymod_hfb}')
plt.legend()
plt.ylabel('streamflow')
plt.xlabel('Time')
plt.title(f'Precip RMSE: {precip_rmse}')
plt.grid(True, linestyle='--', alpha=0.8)
plt.show()
