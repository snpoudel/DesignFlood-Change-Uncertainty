import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmoments3 as lm
from lmoments3 import distr
from scipy.stats import gumbel_r
from scipy.stats import genextreme
from scipy.stats import pearson3

#function that returns flood specific return period
def return_flood(data, return_period, distribution='gev', method='mle'):
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

#read corresponding interpolated precip
noisy_precip_hist = pd.read_csv(f'data/noisy_precip/noisy_precip{id}_coverage{coverage}_comb{comb}.csv') #historical

precip_rmse = rmse(true_precip['PRECIP'], noisy_precip_hist['PRECIP'])
precip_rmse = round(precip_rmse, 2)

#########--TEST FOR HBV/Truth STREAMFLOW--#########
#read true streamflow, historical
true_streamflow = pd.read_csv(f'output/hbv_true/hbv_true{id}.csv')
true_streamflow['year'] = true_streamflow['date'].apply(lambda x: int(x.split('-')[0]))
true_streamflow = true_streamflow.groupby('year')['streamflow'].max()

future_true_streamflow = pd.read_csv(f'output/future/hbv_true/hbv_true{id}.csv')
future_true_streamflow['year'] = future_true_streamflow['date'].apply(lambda x: int(x.split('-')[0]))
future_true_streamflow = future_true_streamflow.groupby('year')['streamflow'].max()
################################################################################################################################################################################################################################################################################################################################################    

#########--TEST FOR HYMOD STREAMFLOW--#########
#read corresponding interpolated hymod streamflow
hymod = pd.read_csv(f'output/simp_hymod/simp_hymod{id}_coverage{coverage}_comb{comb}.csv')
hymod['year'] = hymod['date'].apply(lambda x: int(x.split('-')[0]))
hymod = hymod.groupby('year')['streamflow'].max()

future_hymod = pd.read_csv(f'output/future/simp_hymod/simp_hymod{id}_coverage{coverage}_comb{comb}.csv')
future_hymod['year'] = future_hymod['date'].apply(lambda x: int(x.split('-')[0]))
future_hymod = future_hymod.groupby('year')['streamflow'].max()

#########--TEST FOR LSTM STREAMFLOW--#########
#read corresponding interpolated lstm streamflow
idw_lstm = pd.read_csv(f'output/regional_lstm/historical/lstm_input{id}_coverage{coverage}_comb{comb}.csv')
idw_lstm['year'] = idw_lstm['date'].apply(lambda x: int(x.split('-')[0]))
idw_lstm = idw_lstm.groupby('year')['streamflow'].max()

future_idw_lstm = pd.read_csv(f'output/regional_lstm/future/lstm_input{id}_coverage{coverage}_comb{comb}.csv')
future_idw_lstm['year'] = future_idw_lstm['date'].apply(lambda x: int(x.split('-')[0]))
future_idw_lstm = future_idw_lstm.groupby('year')['streamflow'].max()

##########-TEST FOR HYMOD LSTM STREAMFLOW-#########
#read corresponding interpolated lstm streamflow
simp_hymod_lstm = pd.read_csv(f'output/regional_lstm_simp_hymod/final_output/historical/hymod_lstm{id}_coverage{coverage}_comb{comb}.csv')
simp_hymod_lstm['year'] = simp_hymod_lstm['date'].apply(lambda x: int(x.split('-')[0]))
simp_hymod_lstm = simp_hymod_lstm.groupby('year')['simp_hymod_lstm_streamflow'].max()

future_simp_hymod_lstm = pd.read_csv(f'output/regional_lstm_simp_hymod/final_output/future/hymod_lstm{id}_coverage{coverage}_comb{comb}.csv')
future_simp_hymod_lstm['year'] = future_simp_hymod_lstm['date'].apply(lambda x: int(x.split('-')[0]))
future_simp_hymod_lstm = future_simp_hymod_lstm.groupby('year')['simp_hymod_lstm_streamflow'].max()


# Define return periods
return_periods = np.arange(2, 1041, 10)

# Calculate flood values for each return period for historical condition
true_floods_hist = [return_flood(true_streamflow.values, rp) for rp in return_periods]
hymod_floods_hist = [return_flood(hymod.values, rp) for rp in return_periods]
idw_lstm_floods_hist = [return_flood(idw_lstm.values, rp) for rp in return_periods]
simp_hymod_lstm_floods_hist = [return_flood(simp_hymod_lstm.values, rp) for rp in return_periods]

# Calculate flood values for each return period for future condition
true_floods_future = [return_flood(future_true_streamflow.values, rp) for rp in return_periods]
hymod_floods_future = [return_flood(future_hymod.values, rp) for rp in return_periods]
idw_lstm_floods_future = [return_flood(future_idw_lstm.values, rp) for rp in return_periods]
simp_hymod_lstm_floods_future = [return_flood(future_simp_hymod_lstm.values, rp) for rp in return_periods]

# Function to compute empirical return periods
def empirical_return_periods(data):
    sorted_data = np.sort(data)[::-1]  # Sort in descending order
    n = len(sorted_data)
    empirical_rp = (n + 1) / np.arange(1, n + 1)
    return empirical_rp, sorted_data

# Compute empirical return periods for historical condition
true_empirical_rp_hist, true_empirical_hist = empirical_return_periods(true_streamflow.values)
hymod_empirical_rp_hist, hymod_empirical_hist = empirical_return_periods(hymod.values)
idw_lstm_empirical_rp_hist, idw_lstm_empirical_hist = empirical_return_periods(idw_lstm.values)
simp_hymod_lstm_empirical_rp_hist, simp_hymod_lstm_empirical_hist = empirical_return_periods(simp_hymod_lstm.values)

# Compute empirical return periods for future condition
true_empirical_rp_future, true_empirical_future = empirical_return_periods(future_true_streamflow.values)
hymod_empirical_rp_future, hymod_empirical_future = empirical_return_periods(future_hymod.values)
idw_lstm_empirical_rp_future, idw_lstm_empirical_future = empirical_return_periods(future_idw_lstm.values)
simp_hymod_lstm_empirical_rp_future, simp_hymod_lstm_empirical_future = empirical_return_periods(future_simp_hymod_lstm.values)

# Create plot
fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

# Plot for historical condition
axs[0].plot(return_periods, true_floods_hist, label='True (GEV Fit)', linestyle='-', color='blue')
axs[0].plot(return_periods, hymod_floods_hist, label='Hymod (GEV Fit)', linestyle='-', color='green')
axs[0].plot(return_periods, idw_lstm_floods_hist, label='LSTM (GEV Fit)', linestyle='-', color='orange')
axs[0].plot(return_periods, simp_hymod_lstm_floods_hist, label='Hymod LSTM (GEV Fit)', linestyle='-', color='brown')
axs[0].scatter(true_empirical_rp_hist, true_empirical_hist,  marker='o', s=10, color='blue', alpha=0.6)
axs[0].scatter(hymod_empirical_rp_hist, hymod_empirical_hist,  marker='s', s=10, color='green', alpha=0.6)
axs[0].scatter(idw_lstm_empirical_rp_hist, idw_lstm_empirical_hist,  marker='^', s=10, color='orange', alpha=0.6)
axs[0].scatter(simp_hymod_lstm_empirical_rp_hist, simp_hymod_lstm_empirical_hist,  marker='x', s=10, color='brown', alpha=0.6)
axs[0].set_xscale('log')
axs[0].set_ylabel('Streamflow (mm/day)')
axs[0].set_title(f'Historical Condition - Basin ID: {id}, Precip Error: {precip_rmse}')
axs[0].legend()
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot for future condition
axs[1].plot(return_periods, true_floods_future, label='True (GEV Fit)', linestyle='-', color='blue')
axs[1].plot(return_periods, hymod_floods_future, label='Hymod (GEV Fit)', linestyle='-', color='green')
axs[1].plot(return_periods, idw_lstm_floods_future, label='LSTM (GEV Fit)', linestyle='-', color='orange')
axs[1].plot(return_periods, simp_hymod_lstm_floods_future, label='Hymod LSTM (GEV Fit)', linestyle='-', color='brown')
axs[1].scatter(true_empirical_rp_future, true_empirical_future,  marker='o', s=10, color='blue', alpha=0.6)
axs[1].scatter(hymod_empirical_rp_future, hymod_empirical_future,  marker='s', s=10, color='green', alpha=0.6)
axs[1].scatter(idw_lstm_empirical_rp_future, idw_lstm_empirical_future,  marker='^', s=10, color='orange', alpha=0.6)
axs[1].scatter(simp_hymod_lstm_empirical_rp_future, simp_hymod_lstm_empirical_future, marker='x', s=10, color='brown', alpha=0.6)
axs[1].set_xscale('log')
axs[1].set_xlabel('Return Period (years)')
axs[1].set_ylabel('Streamflow (mm/day)')
axs[1].set_title(f'Future Condition - Basin ID: {id}, Precip Error: {precip_rmse}')
axs[1].legend()
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

#save the plot
fig.savefig(f'output/figures/Return_period{id}.png', dpi=300)