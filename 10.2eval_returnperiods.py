import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyextremes import EVA

#read the list of basin ID
id = '01108000'
coverage = 99 #99 is for truth
comb = 0

#write a function to calculate RMSE
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value
 
#write a function that takes in a pandas series and returns the extreme values
def return_tyr_flood(data, ax=None):
    '''
    Input
    data: It most be a pandas series with datetime index
    Output
    returns the value of the flood for 20, 50 and 100 years return period
    '''
    #create a eva model
    eva_model = EVA(data) #input data must be a pandas series with datetime index
    #find the extreme values
    eva_model.get_extremes(method='BM', extremes_type='high') #Finds 1 extreme value per year
    #visualize the extreme values
    #eva_model.extremes.plot()
    #fit the model
    eva_model.fit_model() # By default, the best fitting distribution is selected using the AIC
    # #calculate the return period
    # eva_summary = eva_model.get_summary(
    #     return_period=[20, 50, 100],
    #     alpha=0.95, #Confidence interval
    #     n_samples=2,) #1000#Number of samples for bootstrap confidence intervals
    # #convert this into a dataframe
    # eva_summary = pd.DataFrame(eva_summary)
    #make a return period plot
    eva_model.plot_return_values(alpha=None, ax=ax) #alpha is the confidence interval
    #model diagnostics plot
    # eva_model.plot_diagnostic(alpha=None) #alpha is the confidence interval
    # plt.show() 
    # return round(eva_summary.iloc[0,0],2), round(eva_summary.iloc[1,0],2), round(eva_summary.iloc[2,0],2)

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
true_streamflow['date'] = pd.to_datetime(true_streamflow['date'])

future_true_streamflow = pd.read_csv(f'output/future/hbv_true_future_streamflow/hbv_true_future_output_{id}.csv')
future_true_streamflow['date'] = pd.to_datetime(true_streamflow['date'])

#return period plot historical
true_flow = true_streamflow.copy()
true_flow.set_index('date', inplace=True)
true_flow.index = pd.to_datetime(true_flow.index, format='%Y-%m-%d')
#return period, future true
future_flow = future_true_streamflow.copy()
future_flow.set_index('date', inplace=True)
future_flow.index = pd.to_datetime(future_flow.index, format='%Y-%m-%d')


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
