#import libraries
import numpy as np
import pandas as pd
from pyextremes import EVA #https://georgebv.github.io/pyextremes/quickstart/

#read data
basin_id = '01108000'
grid = 0.05
comb = 5
hbv_true = pd.read_csv(f'output/hbv_idw_streamflow/hbv_idw_streamflow{basin_id}_coverage{grid}_comb{comb}.csv', index_col=1)
hbv_true.index = pd.to_datetime(hbv_true.index, format='%Y-%m-%d')
hbv_true_flow = hbv_true['streamflow'] #select the streamflow data
hbv_true_flow = pd.Series(hbv_true_flow) #convert to pandas series

#create a eva model
eva_model = EVA(hbv_true_flow) #input data must be a pandas series with datetime index
#find the extreme values
eva_model.get_extremes(method='BM', extremes_type='high', block_size='365D') #Finds 1 extreme value per year
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


#read data
basin_id = '01108000'
grid = 0.05
comb = 1
hbv_recalibrated = pd.read_csv(f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{basin_id}_coverage{grid}_comb{comb}.csv', index_col=1)
hbv_recalibrated.index = pd.to_datetime(hbv_recalibrated.index, format='%Y-%m-%d')
hbv_recalibrated_flow = hbv_recalibrated['streamflow']
hbv_recalibrated_flow = pd.Series(hbv_recalibrated_flow)

#create a eva model
eva_model = EVA(hbv_recalibrated_flow) #input data must be a pandas series with datetime index
#find the extreme values
eva_model.get_extremes(method='BM', extremes_type='high', block_size='365D') #Finds 1 extreme value per year
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


#read data
basin_id = '01108000'
grid = 0.05
comb = 5
hymod = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{basin_id}_coverage{grid}_comb{comb}.csv', index_col=1)
hymod.index = pd.to_datetime(hymod.index, format='%Y-%m-%d')
hymod_flow = hymod['streamflow']
hymod_flow = pd.Series(hymod_flow)

#create a eva model
eva_model = EVA(hymod_flow) #input data must be a pandas series with datetime index
#find the extreme values
eva_model.get_extremes(method='BM', extremes_type='high', block_size='365D') #Finds 1 extreme value per year
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