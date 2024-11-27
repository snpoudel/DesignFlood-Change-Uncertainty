import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmoments3 as lm
from lmoments3 import distr
from scipy.stats import gumbel_r
from scipy.stats import genextreme
from scipy.stats import pearson3

#function that returns flood specific return period
def return_flood(data, return_period, distribution, method):
    '''
    data: annual extreme values, eg., [10, 20, 30]
    return_period: 5-yr, eg., 5
    distribution: Gumbel = 'gum' Log-Pearson3 =  'lp' GEV = 'gev'
    method: L-moment = 'lm' or MLE = 'mle'
    '''
    data = data[data>0] #only keep non zero values
    #calculate non exceedance probability from return period
    exceedance_prob = 1/return_period
    nep = 1 - exceedance_prob #non-exceedance probability

    if distribution == 'gum': ##--Gumbel Distribution--##
        
        if method == 'lm': #fit using L-moment
            params = distr.gum.lmom_fit(data)
            model = distr.gum(**params)
            flood = model.ppf(nep)
            return flood
        if method == 'mle': #fit using MLE
            params = gumbel_r.fit(data) #MLE is default
            flood = gumbel_r.ppf(nep, loc=params[0], scale=params[1])
            return flood
        
    if distribution == 'gev': ##--Generalized Extreme Value distribution--##
        
        if method == 'lm':
            #fit with L-moment
            params = distr.gev.lmom_fit(data)
            model = distr.gev(**params)
            flood = model.ppf(nep)
            return flood
        if method == 'mle': #fit with MLE
            params = genextreme.fit(data)
            flood = genextreme.ppf(nep, c=params[0], loc=params[1], scale=params[2])
            return flood
        
    if distribution == 'lp': ##--Log-Normal distribution--##
        
        if method == 'lm':
            #fit with L-moment
            params = distr.pe3.lmom_fit(np.log(data))
            model = distr.pe3(**params)
            flood = np.exp(model.ppf(nep))
            return flood
        if method == 'mle':
            #fit with MLE
            params = pearson3.fit(np.log(data))
            flood = np.exp(pearson3.ppf(nep, skew=params[0], loc=params[1], scale=params[2]))
            return flood


#read the list of basin ID
id = '01108000'
coverage = 99 #99 is for truth
comb = 0

#write a function to calculate RMSE
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value
 

#########--TEST FOR PRECIPTATION--######### historical and future
#read true precip, historical
true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')
true_precip['time'] = pd.to_datetime(true_precip['DATE'])
#read true precip, historical
future_true_precip = pd.read_csv(f'data/future/future_true_precip/future_true_precip{id}.csv')
future_true_precip['time'] = pd.to_datetime(true_precip['DATE'])
#read corresponding interpolated precip
idw_precip_hist = pd.read_csv(f'data/idw_precip/idw_precip{id}_coverage{coverage}_comb{comb}.csv') #historical
idw_precip_future= pd.read_csv(f'data/future/future_idw_precip/future_idw_precip{id}_coverage{coverage}_comb{comb}.csv') #future

precip_rmse = rmse(true_precip['PRECIP'], idw_precip_hist['PRECIP'])
precip_rmse = round(precip_rmse, 2)

#########--TEST FOR HBV/Truth STREAMFLOW--#########
#read true streamflow, historical
true_streamflow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
true_streamflow['year'] = pd.to_datetime(true_streamflow['date']).dt.year
data = true_streamflow.groupby('year')['streamflow'].max()

future_true_streamflow = pd.read_csv(f'output/future/hbv_true_future_streamflow/hbv_true_future_output_{id}.csv')
future_true_streamflow['year'] = pd.to_datetime(true_streamflow['date']).dt.year
data_future = future_true_streamflow.groupby('year')['streamflow'].max()


            





#########--TEST FOR HYMOD STREAMFLOW--#########
#read corresponding interpolated hymod streamflow
idw_hymod = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{id}_coverage{coverage}_comb{comb}.csv')
future_idw_hymod = pd.read_csv(f'output/future/hymod_idw_future_streamflow/hymod_interpol_future_streamflow{id}_coverage{coverage}_comb{comb}.csv')
#return period plot historical
idw_hymod_flow = idw_hymod.copy()
idw_hymod_flow.set_index('date', inplace=True)
idw_hymod_flow.index = pd.to_datetime(idw_hymod_flow.index, format='%Y-%m-%d')
#return period, future hymod
future_idw_hymod_flow = future_idw_hymod.copy()
future_idw_hymod_flow.set_index('date', inplace=True)
future_idw_hymod_flow.index = pd.to_datetime(future_idw_hymod_flow.index, format='%Y-%m-%d')

#########--TEST FOR LSTM STREAMFLOW--#########
#read corresponding interpolated lstm streamflow
idw_lstm = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage{coverage}_comb{comb}.csv')
future_idw_lstm = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}_coverage{coverage}_comb{comb}.csv')
#return period plot historical
idw_lstm_flow = idw_lstm.copy()
idw_lstm_flow.set_index('date', inplace=True)
idw_lstm_flow.index = pd.to_datetime(idw_lstm_flow.index, format='%Y-%m-%d')
#return period, future lstm
future_idw_lstm_flow = future_idw_lstm.copy()
future_idw_lstm_flow.set_index('date', inplace=True)
future_idw_lstm_flow.index = pd.to_datetime(future_idw_lstm_flow.index, format='%Y-%m-%d')


#plot y axis from mm/day to m3/s
drainage_area = 677.6 #sqkm
# drainage_area = drainage_area*86400/(1000*677.6) #delete me later
#convert mm/day to m3/s and make the plot
fig, axs = plt.subplots(3,1,figsize=(6,8), sharex=True)
return_tyr_flood(true_flow['streamflow']*drainage_area*1000/86400, ax=axs[0])
return_tyr_flood(future_flow['streamflow']*drainage_area*1000/86400, ax=axs[0])
axs[0].set_ylabel(f'"Truth" Streamflow (m3/s)')
axs[0].set_xlabel('Return Period (years)')
return_tyr_flood(idw_hymod_flow['streamflow']*drainage_area*1000/86400, ax=axs[1])
return_tyr_flood(future_idw_hymod_flow['streamflow']*drainage_area*1000/86400, ax=axs[1])
axs[1].set_ylabel('Hymod Streamflow (m3/s)')
axs[1].set_xlabel('Return Period (years)')
return_tyr_flood(idw_lstm_flow['streamflow']*drainage_area*1000/86400, ax=axs[2])
return_tyr_flood(future_idw_lstm_flow['streamflow']*drainage_area*1000/86400, ax=axs[2])
axs[2].set_ylabel('LSTM Streamflow (m3/s)')
axs[2].set_xlabel('Return Period (years)')
plt.suptitle(f'Basin ID: {id}, Precip error: {precip_rmse}')
plt.tight_layout()
plt.show()
# #save the plot
fig.savefig(f'output/figures/{id}/Return_period{id}.png', dpi=300)
# #also save as svg
# fig.savefig(f'inkscape/cumecs_return_period{id}.svg', dpi=300)
