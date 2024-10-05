#import libraries
import numpy as np
import pandas as pd
import os
from pyextremes import EVA #https://georgebv.github.io/pyextremes/quickstart/

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
        return_period=[5, 10, 20],
        alpha=0.95, #Confidence interval
        n_samples=2,) #1000#Number of samples for bootstrap confidence intervals
    #convert this into a dataframe
    eva_summary = pd.DataFrame(eva_summary)
    #model diagnostics plot
    #eva_model.plot_diagnostic(alpha=None) #alpha is the confidence interval
    #return the value of the flood for 20, 50 and 100 years return period
    return eva_summary.iloc[0,0], eva_summary.iloc[1,0], eva_summary.iloc[2,0]


# basin_id = '01109060'
used_basin_list = ['01108000', '01109060', '01177000', '01104500']
for basin_id in used_basin_list:
    df_tyr_flood =pd.DataFrame(columns=['model', 'grid', 'comb', '5yr_flood', '10yr_flood', '20yr_flood', 'precip_rmse'])
    df_change_flood = pd.DataFrame(columns=['model', 'change_5yr_flood', 'change_10yr_flood', 'change_20yr_flood', 'precip_rmse'])

    #True precipitation
    #read true precipitation
    true_precip = pd.read_csv(f'data/true_precip/true_precip{basin_id}.csv')
    future_true_precip = pd.read_csv(f'data/future/future_true_precip/future_true_precip{basin_id}.csv')

    #HBV truth model
    #read the streamflow data
    hbv_truth = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{basin_id}.csv', index_col=0)
    hbv_truth.index = pd.to_datetime(hbv_truth.index, format='%Y-%m-%d')
    hbv_truth_flow = hbv_truth['streamflow'] #select the streamflow data
    hbv_truth_flow = pd.Series(hbv_truth_flow) #convert to pandas series
    #calculate the 20, 50 and 100 years flood
    flood_5, flood_10, flood_20 = return_tyr_flood(hbv_truth_flow)
    true_tyr_flood =pd.DataFrame({'model':'HBV True', 'grid':'NA', 'comb':'NA', '5yr_flood':flood_5, '10yr_flood':flood_10, '20yr_flood':flood_20, 'precip_rmse':0}, index=[0])
    df_tyr_flood = pd.concat([df_tyr_flood, true_tyr_flood], ignore_index=True)

    #read the future streamflow data
    hbv_true_future = pd.read_csv(f'output/future/hbv_true_future_streamflow/hbv_true_future_output_{basin_id}.csv', index_col=0)
    hbv_true_future.index = pd.to_datetime(hbv_true_future.index, format='%Y-%m-%d')
    hbv_true_future_flow = hbv_true_future['streamflow'] #select the streamflow data
    hbv_true_future_flow = pd.Series(hbv_true_future_flow) #convert to pandas series
    #calculate the 20, 50 and 100 years flood
    flood_5, flood_10, flood_20 = return_tyr_flood(hbv_true_future_flow)
    true_tyr_flood_future =pd.DataFrame({'model':'HBV True Future', 'grid':'NA', 'comb':'NA', '5yr_flood':flood_5, '10yr_flood':flood_10, '20yr_flood':flood_20, 'precip_rmse':0}, index=[0])
    df_tyr_flood = pd.concat([df_tyr_flood, true_tyr_flood_future], ignore_index=True)

    #calculate the change in flood for true model
    change_hbv_true = pd.DataFrame({'model':'HBV True', 'change_5yr_flood':(true_tyr_flood_future['5yr_flood'] - true_tyr_flood['5yr_flood']),
                        'change_10yr_flood':(true_tyr_flood_future['10yr_flood'] - true_tyr_flood['10yr_flood']),
                        'change_20yr_flood':(true_tyr_flood_future['20yr_flood'] - true_tyr_flood['20yr_flood']), 'precip_rmse':0}, index=[0])
    df_change_flood = pd.concat([df_change_flood, change_hbv_true], ignore_index=True)


    #loop through each grid coverage and combination
    grid_list = np.arange(30)
    grid_list = np.append(grid_list, [99])
    for grid in grid_list:
        for comb in range(15):
            #--HISTORICAL DATA--#
            #Interpolated precipitation
            file_path = f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{basin_id}_coverage{grid}_comb{comb}.csv'
            if os.path.exists(file_path):
                idw_precip = pd.read_csv(f'data/idw_precip/idw_precip{basin_id}_coverage{grid}_comb{comb}.csv')
                future_idw_precip = pd.read_csv(f'data/future/future_idw_precip/future_idw_precip{basin_id}_coverage{grid}_comb{comb}.csv')
                precip_rmse = np.sqrt(np.mean((true_precip['PRECIP'] - idw_precip['PRECIP'])**2)) #calculate the rmse
                if np.array_equal(true_precip['PRECIP'], idw_precip['PRECIP']):
                    rmse_precip = 0
                precip_rmse_future = np.sqrt(np.mean((future_true_precip['PRECIP'] - future_idw_precip['PRECIP'])**2))
        
                #HBV true model
                #read the streamflow data
                # if os.path.exists(f'output/hbv_idw_streamflow/hbv_idw_streamflow{basin_id}_coverage{grid}_comb{comb}.csv'):
                hbv_true = pd.read_csv(f'output/hbv_idw_streamflow/hbv_idw_streamflow{basin_id}_coverage{grid}_comb{comb}.csv', index_col=1)
                hbv_true.index = pd.to_datetime(hbv_true.index, format='%Y-%m-%d')
                hbv_true_flow = hbv_true['streamflow'] #select the streamflow data
                hbv_true_flow = pd.Series(hbv_true_flow) #convert to pandas series
                #calculate the 20, 50 and 100 years flood
                flood_5, flood_10, flood_20 = return_tyr_flood(hbv_true_flow)
                temp_df_hbv =pd.DataFrame({'model':'HBV True', 'grid':grid, 'comb':comb, '5yr_flood':flood_5, '10yr_flood':flood_10, '20yr_flood':flood_20, 'precip_rmse':precip_rmse}, index=[0])
                df_tyr_flood = pd.concat([df_tyr_flood, temp_df_hbv], ignore_index=True)

                #HBV recalibrated model
                #read the streamflow data
                hbv_recalibrated = pd.read_csv(f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{basin_id}_coverage{grid}_comb{comb}.csv', index_col=1)
                hbv_recalibrated.index = pd.to_datetime(hbv_recalibrated.index, format='%Y-%m-%d')
                hbv_recalibrated_flow = hbv_recalibrated['streamflow']
                hbv_recalibrated_flow = pd.Series(hbv_recalibrated_flow)
                #calculate the 20, 50 and 100 years flood
                flood_5, flood_10, flood_20 = return_tyr_flood(hbv_recalibrated_flow)
                temp_df_hbvr = pd.DataFrame({'model':'HBV Recalib', 'grid':grid, 'comb':comb, '5yr_flood':flood_5, '10yr_flood':flood_10, '20yr_flood':flood_20, 'precip_rmse':precip_rmse}, index=[0])
                df_tyr_flood = pd.concat([df_tyr_flood, temp_df_hbvr], ignore_index=True)

                #Hymod model
                #read the streamflow data
                hymod = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{basin_id}_coverage{grid}_comb{comb}.csv', index_col=1)
                hymod.index = pd.to_datetime(hymod.index, format='%Y-%m-%d')
                hymod_flow = hymod['streamflow']
                hymod_flow = pd.Series(hymod_flow)
                #calculate the 20, 50 and 100 years flood
                flood_5, flood_10, flood_20 = return_tyr_flood(hymod_flow)
                temp_df_hy = pd.DataFrame({'model':'Hymod', 'grid':grid, 'comb':comb, '5yr_flood':flood_5, '10yr_flood':flood_10, '20yr_flood':flood_20, 'precip_rmse':precip_rmse}, index=[0])
                df_tyr_flood = pd.concat([df_tyr_flood, temp_df_hy], ignore_index=True)

                #LSTM model
                #read the streamflow data
                if os.path.exists(f'output/regional_lstm/historical/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv'):
                    lstm = pd.read_csv(f'output/regional_lstm/historical/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv', index_col=0)
                    lstm.index = pd.to_datetime(lstm.index, format='%Y-%m-%d')
                    lstm_flow = lstm['streamflow']
                    lstm_flow = pd.Series(lstm_flow)
                    #calculate the 20, 50 and 100 years flood
                    flood_5, flood_10, flood_20 = return_tyr_flood(lstm_flow)
                else:
                    flood_5, flood_10, flood_20 = np.NAN, np.NAN, np.NAN

                temp_df_lstm = pd.DataFrame({'model':'LSTM', 'grid':grid, 'comb':comb, '5yr_flood':flood_5, '10yr_flood':flood_10, '20yr_flood':flood_20, 'precip_rmse':precip_rmse}, index=[0])
                df_tyr_flood = pd.concat([df_tyr_flood, temp_df_lstm], ignore_index=True)

                #--FUTURE DATA--#
                #HBV true model future
                #read the streamflow data
                # if os.path.exists(f'output/future/hbv_idw_future_streamflow/hbv_idw_future_streamflow{basin_id}_coverage{grid}_comb{comb}.csv'):
                hbv_true_future = pd.read_csv(f'output/future/hbv_idw_future_streamflow/hbv_idw_future_streamflow{basin_id}_coverage{grid}_comb{comb}.csv', index_col=1)
                hbv_true_future.index = pd.to_datetime(hbv_true_future.index, format='%Y-%m-%d')
                hbv_true_future_flow = hbv_true_future['streamflow'] #select the streamflow data
                hbv_true_future_flow = pd.Series(hbv_true_future_flow) #convert to pandas series
                #calculate the 20, 50 and 100 years flood
                flood_5, flood_10, flood_20 = return_tyr_flood(hbv_true_future_flow)
                temp_df_hbvf =pd.DataFrame({'model':'HBV True Future', 'grid':grid, 'comb':comb, '5yr_flood':flood_5, '10yr_flood':flood_10, '20yr_flood':flood_20, 'precip_rmse':precip_rmse}, index=[0])
                df_tyr_flood = pd.concat([df_tyr_flood, temp_df_hbvf], ignore_index=True)

                #HBV recalibrated model future
                #read the streamflow data
                hbv_recalibrated_future =pd.read_csv(f'output/future/hbv_idw_recalib_future_streamflow/hbv_idw_recalib_future_streamflow{basin_id}_coverage{grid}_comb{comb}.csv', index_col=1)
                hbv_recalibrated_future.index = pd.to_datetime(hbv_recalibrated_future.index, format='%Y-%m-%d')
                hbv_recalibrated_future_flow = hbv_recalibrated_future['streamflow']
                hbv_recalibrated_future_flow = pd.Series(hbv_recalibrated_future_flow)
                #calculate the 20, 50 and 100 years flood
                flood_5, flood_10, flood_20 = return_tyr_flood(hbv_recalibrated_future_flow)
                temp_df_hbvrf = pd.DataFrame({'model':'HBV Recalib Future', 'grid':grid, 'comb':comb, '5yr_flood':flood_5, '10yr_flood':flood_10, '20yr_flood':flood_20, 'precip_rmse':precip_rmse}, index=[0])
                df_tyr_flood = pd.concat([df_tyr_flood, temp_df_hbvrf], ignore_index=True)

                #Hymod model future
                #read the streamflow data
                hymod_future = pd.read_csv(f'output/future/hymod_idw_future_streamflow/hymod_interpol_future_streamflow{basin_id}_coverage{grid}_comb{comb}.csv', index_col=1)
                hymod_future.index = pd.to_datetime(hymod_future.index, format='%Y-%m-%d')
                hymod_future_flow = hymod_future['streamflow']
                hymod_future_flow = pd.Series(hymod_future_flow)
                #calculate the 20, 50 and 100 years flood
                flood_5, flood_10, flood_20 = return_tyr_flood(hymod_future_flow)
                temp_df_hyf = pd.DataFrame({'model':'Hymod Future', 'grid':grid, 'comb':comb, '5yr_flood':flood_5, '10yr_flood':flood_10, '20yr_flood':flood_20, 'precip_rmse':precip_rmse}, index=[0])
                df_tyr_flood = pd.concat([df_tyr_flood, temp_df_hyf], ignore_index=True)

                #LSTM model future
                #read the streamflow data
                if os.path.exists(f'output/regional_lstm/future/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv'):
                    lstm_future = pd.read_csv(f'output/regional_lstm/future/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv', index_col=0)
                    lstm_future.index = pd.to_datetime(lstm_future.index, format='%Y-%m-%d')
                    lstm_future_flow = lstm_future['streamflow']
                    lstm_future_flow = pd.Series(lstm_future_flow)
                    #calculate the 20, 50 and 100 years flood
                    flood_5, flood_10, flood_20 = return_tyr_flood(lstm_future_flow)
                else:
                    flood_5, flood_10, flood_20 = np.NaN, np.NaN, np.NaN
                temp_df_lstmf = pd.DataFrame({'model':'LSTM Future', 'grid':grid, 'comb':comb, '5yr_flood':flood_5, '10yr_flood':flood_10, '20yr_flood':flood_20, 'precip_rmse':precip_rmse}, index=[0])
                df_tyr_flood = pd.concat([df_tyr_flood, temp_df_lstmf], ignore_index=True)

                #--CHANGE IN FLOOD (FUTURE - HISTORICAL)--#
                change_hbv_true = pd.DataFrame({'model':'HBV True', 'change_5yr_flood':(temp_df_hbvf['5yr_flood'] - temp_df_hbv['5yr_flood']),
                                'change_10yr_flood':(temp_df_hbvf['10yr_flood'] - temp_df_hbv['10yr_flood']),
                                'change_20yr_flood':(temp_df_hbvf['20yr_flood'] - temp_df_hbv['20yr_flood']), 'precip_rmse':precip_rmse}, index=[0])

                change_hbv_recalib = pd.DataFrame({'model':'HBV Recalib', 'change_5yr_flood':(temp_df_hbvrf['5yr_flood'] - temp_df_hbvr['5yr_flood']),
                                'change_10yr_flood':(temp_df_hbvrf['10yr_flood'] - temp_df_hbvr['10yr_flood']),
                                'change_20yr_flood':(temp_df_hbvrf['20yr_flood'] - temp_df_hbvr['20yr_flood']), 'precip_rmse':precip_rmse}, index=[0])
                
                change_hymod = pd.DataFrame({'model':'Hymod', 'change_5yr_flood':(temp_df_hyf['5yr_flood'] - temp_df_hy['5yr_flood']),
                                'change_10yr_flood':(temp_df_hyf['10yr_flood'] - temp_df_hy['10yr_flood']),
                                'change_20yr_flood':(temp_df_hyf['20yr_flood'] - temp_df_hy['20yr_flood']), 'precip_rmse':precip_rmse}, index=[0])
                
                change_lstm = pd.DataFrame({'model':'LSTM', 'change_5yr_flood':(temp_df_lstmf['5yr_flood'] - temp_df_lstm['5yr_flood']),
                                'change_10yr_flood':(temp_df_lstmf['10yr_flood'] - temp_df_lstm['10yr_flood']),
                                'change_20yr_flood':(temp_df_lstmf['20yr_flood'] - temp_df_lstm['20yr_flood']), 'precip_rmse':precip_rmse}, index=[0])
                
                df_change_flood = pd.concat([df_change_flood, change_hbv_true, change_hbv_recalib, change_hymod, change_lstm], ignore_index=True)

    #save the dataframes
    df_tyr_flood.to_csv(f'output/tyr_flood_{basin_id}.csv', index=False)
    df_change_flood.to_csv(f'output/change_tyr_flood_{basin_id}.csv', index=False)