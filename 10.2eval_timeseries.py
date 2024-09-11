import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyextremes import EVA

#read the list of basin ID
id = '01108000'
coverage = 10
comb = 9 #3

#plot window
plot_start = 2000 #2165start at least from second year to match with LSTM streamflow
plot_end = 2300 #2515

#write a function to calculate RMSE
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value
 
#write a function that takes in a pandas series and returns the extreme values
def return_tyr_flood(data):
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
    #calculate the return period
    eva_summary = eva_model.get_summary(
        return_period=[20, 50, 100],
        alpha=0.95, #Confidence interval
        n_samples=2,) #1000#Number of samples for bootstrap confidence intervals
    #convert this into a dataframe
    eva_summary = pd.DataFrame(eva_summary)
    #model diagnostics plot
    eva_model.plot_diagnostic(alpha=None) #alpha is the confidence interval
    #return the value of the flood for 20, 50 and 100 years return period
    #print(eva_summary.iloc[0,0], eva_summary.iloc[1,0], eva_summary.iloc[2,0])
    plt.show() 
    return round(eva_summary.iloc[0,0],2), round(eva_summary.iloc[1,0],2), round(eva_summary.iloc[2,0],2)
   

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

#########--TEST FOR HBV STREAMFLOW--#########
#read true streamflow, historical
true_streamflow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
true_streamflow['date'] = pd.to_datetime(true_streamflow['date'])

future_true_streamflow = pd.read_csv(f'output/future/hbv_true_future_streamflow/hbv_true_future_output_{id}.csv')
future_true_streamflow['date'] = pd.to_datetime(true_streamflow['date'])

#########--TEST FOR HYMOD STREAMFLOW--#########
#read corresponding interpolated hymod streamflow
idw_hymod = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{id}_coverage{coverage}_comb{comb}.csv')
future_idw_hymod = pd.read_csv(f'output/future/hymod_idw_future_streamflow/hymod_interpol_future_streamflow{id}_coverage{coverage}_comb{comb}.csv')

#########--TEST FOR HBV RECALIBRATED STREAMFLOW--#########
#read corresponding interpolated hymod streamflow
idw_hbvrecal = pd.read_csv(f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{id}_coverage{coverage}_comb{comb}.csv')
future_idw_hbvrecal = pd.read_csv(f'output/future/hbv_idw_recalib_future_streamflow/hbv_idw_recalib_future_streamflow{id}_coverage{coverage}_comb{comb}.csv')


# #plot the true and interpolated precip
# fig, axes = plt.subplots(2,1, figsize = (12,10), sharex=True)
# axes[0].plot(true_precip['time'][plot_start:plot_end], true_precip['PRECIP'][plot_start:plot_end], color = 'blue')
# axes[0].plot(future_true_precip['time'][plot_start:plot_end], future_true_precip['PRECIP'][plot_start:plot_end], color ='orange')
# axes[0].plot(true_precip['time'][plot_start:plot_end], idw_precip_hist['PRECIP'][plot_start:plot_end], linestyle = '--', color = 'blue')
# axes[0].plot(true_precip['time'][plot_start:plot_end], idw_precip_future['PRECIP'][plot_start:plot_end], linestyle = '--', color = 'orange')
# axes[0].set_ylabel("Precipitation(mm/day)")
# axes[0].set_title(f'No. of gauge used: {coverage}   Precip RMSE: {precip_rmse}')

# #plot the true and interpolated streamflow
# axes[1].plot(true_streamflow['date'][plot_start:plot_end], true_streamflow['streamflow'][plot_start:plot_end], color = 'blue')
# axes[1].plot(future_true_streamflow['date'][plot_start:plot_end], future_true_streamflow['streamflow'][plot_start:plot_end], color = 'orange')
# axes[1].plot(true_streamflow['date'][plot_start:plot_end], idw_hymod['streamflow'][plot_start:plot_end], linestyle = '--', color = 'blue')
# axes[1].plot(true_streamflow['date'][plot_start:plot_end], future_idw_hymod['streamflow'][plot_start:plot_end], linestyle = '--', color = 'orange')
# axes[1].legend(['Truth Hist', 'Truth Future', 'Hymod Hist', 'Hymod Future'])
# axes[1].set_xlabel("Date")
# axes[1].set_ylabel("Streamflow(mm/day)")
# plt.tight_layout()
# plt.show()
# #save the plot
# fig.savefig('output/eval_streamflow_timeseries.png', dpi = 300)



# #plot historical streamflows only
# fig,axes =plt.subplots(2,1, figsize=(8,8), sharey=True)
# axes[0].plot(true_streamflow['date'], true_streamflow['streamflow'],  label='Truth')
# axes[0].legend()
# axes[0].set_title('Historical')
# axes[1].plot(true_streamflow['date'], idw_hymod['streamflow'],  label='Hymod')
# axes[1].legend()


# #plot future precipitations
# fig,axes =plt.subplots(2,1, figsize=(8,8), sharey=True)
# axes[0].plot(future_true_precip['DATE'], future_true_precip['PRECIP'],  label='Truth')
# axes[0].legend()
# axes[0].set_title('Future')
# axes[1].plot(idw_precip_future['DATE'], idw_precip_future['PRECIP'],  label='IDW')
# axes[1].legend()
# plt.show()

#return period, future true
future_flow = future_true_streamflow.copy()
future_flow.set_index('date', inplace=True)
future_flow.index = pd.to_datetime(future_flow.index, format='%Y-%m-%d')
true20,true50,true100 = return_tyr_flood(future_flow['streamflow'])

#return period, future hymod
future_idw_hymod_flow = future_idw_hymod.copy()
future_idw_hymod_flow.set_index('date', inplace=True)
future_idw_hymod_flow.index = pd.to_datetime(future_idw_hymod_flow.index, format='%Y-%m-%d')
idw20,idw50,idw100 = return_tyr_flood(future_idw_hymod_flow['streamflow'])

#return period, future hbv recalibrated
future_idw_hbvrecalib_flow = future_idw_hbvrecal.copy()
future_idw_hbvrecalib_flow.set_index('date', inplace=True)
future_idw_hbvrecalib_flow.index = pd.to_datetime(future_idw_hbvrecalib_flow.index, format='%Y-%m-%d')
ridw20,ridw50,ridw100 = return_tyr_flood(future_idw_hbvrecalib_flow['streamflow'])


#plot future streamflows only
fig,axes =plt.subplots(3,1, figsize=(10,10), sharey=True)
axes[0].plot(true_streamflow['date'], future_true_streamflow['streamflow'],  label=f'Future Truth\n 20yr-flood={true20}\n 50yr-flood={true50}')
axes[0].legend()
axes[0].set_title(f'Precip RMSE: 0')
axes[0].grid(True, linestyle='--', alpha=0.8)
axes[1].plot(true_streamflow['date'], future_idw_hbvrecal['streamflow'],  label=f'Future Hbv Recal\n 20yr-flood={ridw20}\n 50yr-flood={ridw50}')
axes[1].legend()
axes[1].set_title(f'Num of gauge used:{coverage}   Precip RMSE:{precip_rmse}')
axes[1].grid(True, linestyle='--', alpha=0.8)
axes[2].plot(true_streamflow['date'], future_idw_hymod['streamflow'],  label=f'Future Hymod\n 20yr-flood={idw20}\n 50yr-flood={idw50}')
axes[2].legend()
axes[2].set_title(f'Num of gauge used:{coverage}   Precip RMSE:{precip_rmse}')
axes[2].grid(True, linestyle='--', alpha=0.8)
plt.tight_layout()
plt.show()

