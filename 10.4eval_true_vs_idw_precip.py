import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#write a function to calculate RMSE
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value

#write a function to calculate Percentage BIAS
def pbias(q_obs, q_sim):
    #pbias_value = (q_obs - q_sim) / np.sum(q_obs) * 100
    pbias_value = q_obs-q_sim
    return pbias_value

#read list of basins
basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv',dtype={'basin_id':str})
id = '01108000'

#quantiles of interest
quant1=0.95
quant2=0.99
quant3=0.999

#read true precipitation
true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')
t999, t9995, t9999 =np.quantile(true_precip['PRECIP'], [quant1,quant2,quant3])

#read true future precipitation
future_true_precip = pd.read_csv(f'data/future/future_true_precip/future_true_precip{id}.csv')
f999, f9995, f9999 =np.quantile(future_true_precip['PRECIP'], [quant1,quant2,quant3])

#read interpolated precipitaiton
df = pd.DataFrame() #empty dataframe to store results
for coverage in np.arange(30):
    for comb in np.arange(15):
        file_path = f'data/idw_precip/idw_precip{id}_coverage{coverage}_comb{comb}.csv'
        if os.path.exists(file_path):
            idw_precip = pd.read_csv(file_path) #historical
            future_idw_precip=pd.read_csv(f'data/future/future_idw_precip/future_idw_precip{id}_coverage{coverage}_comb{comb}.csv')
            precip_rmse = rmse(true_precip['PRECIP'], idw_precip['PRECIP'])
            #calculate qunatiles for idw precip
            i999, i9995, i9999 =np.quantile(idw_precip['PRECIP'], [quant1,quant2,quant3])
            #historical difference
            hist_999 = pbias(t999, i999)
            hist_9995 = pbias(t9995, i9995)
            hist_9999 = pbias(t9999, i9999)

            #calculate qunatiles for future idw precip
            fi999, fi9995, fi9999 =np.quantile(future_idw_precip['PRECIP'], [quant1,quant2,quant3])
            #future difference
            future_999 = pbias(f999, fi999)
            future_9995 = pbias(f9995, fi9995)
            future_9999 = pbias(f9999, fi9999)

            #change
            change_999 = ((i999 - fi999)/i999)*100
            change_9995 = ((i9995 - fi9995)/i9995)*100
            change_9999 = ((i9999 - fi9999)/i9999)*100

            #save into a dataframe
            temp_df = pd.DataFrame({'precip_rmse':precip_rmse, 'hist_999':hist_999, 'hist_9995':hist_9995, 'hist_9999':hist_9999,
                                    'future_999':future_999, 'future_9995':future_9995, 'future_9999':future_9999,
                                    'change_999':change_999,'change_9995':change_9995, 'change_9999':change_9999 },index=[0])
            #merge to a dataframe
            df = pd.concat([df, temp_df], ignore_index=True)
           

#plots
fig,axs = plt.subplots(3,1, figsize=(8,9), sharex=True)
#historical
sns.regplot(x='precip_rmse', y='hist_999', lowess=True, label=f'{quant1}th precip', data=df, ax=axs[0])
sns.regplot(x='precip_rmse', y='hist_9995', lowess=True, label=f'{quant2}th precip', data=df, ax=axs[0])
sns.regplot(x='precip_rmse', y='hist_9999', lowess=True, label=f'{quant3}th precip', data=df, ax=axs[0])
axs[0].set_ylabel('Historical MAP Bias mm/day\n(True - IDW)')
axs[0].set_xlabel('')
axs[0].legend(loc='lower left')
axs[0].grid(True, linestyle='--',alpha=0.5)

#future
sns.regplot(x='precip_rmse', y='future_999', lowess=True, label=f'{quant1}th precip', data=df, ax=axs[1])
sns.regplot(x='precip_rmse', y='future_9995', lowess=True, label=f'{quant2}th precip', data=df, ax=axs[1])
sns.regplot(x='precip_rmse', y='future_9999', lowess=True, label=f'{quant3}th precip', data=df, ax=axs[1])
axs[1].set_ylabel('Future MAP Bias mm/day\n(True - IDW)')
axs[1].set_xlabel('')
#axs[1].legend()
axs[1].grid(True, linestyle='--',alpha=0.5)


#change in MAP
sns.regplot(x='precip_rmse', y='change_999', lowess=True, label=f'{quant1}th precip', data=df, ax=axs[2])
sns.regplot(x='precip_rmse', y='change_9995', lowess=True, label=f'{quant2}th precip', data=df, ax=axs[2])
sns.regplot(x='precip_rmse', y='change_9999', lowess=True, label=f'{quant3}th precip', data=df, ax=axs[2])
axs[2].set_ylabel('Change in MAP %\n(IDW Hist - IDW Future) / IDW Hist')
axs[2].set_xlabel('Mean Areal Precipitation(MAP) RMSE')
# axs[2].legend()
axs[2].grid(True, linestyle='--',alpha=0.5)
plt.tight_layout()
plt.show()

fig.savefig('output/precip99.png', dpi=300)