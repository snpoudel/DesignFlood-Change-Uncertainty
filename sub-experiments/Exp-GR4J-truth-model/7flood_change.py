#import libraries
import numpy as np
import pandas as pd
import os
from scipy.stats import genextreme
import warnings
warnings.filterwarnings('ignore')

# scenario = ['scenario15']
scenario = ['scenario3', 'scenario7', 'scenario11', 'scenario15']

#function that returns flood specific return period
def return_flood(data, return_period, distribution, method):
    data = data[data>0] #only keep non zero values
    #calculate non exceedance probability from return period
    exceedance_prob = 1/return_period
    nep = 1 - exceedance_prob #non-exceedance probability
        
    if distribution == 'gev': ##--Generalized Extreme Value distribution--##
        
        if method == 'mle': #fit with MLE
            params = genextreme.fit(data)
            flood = genextreme.ppf(nep, c=params[0], loc=params[1], scale=params[2])
            return flood


#function to return change in design flood wrt true change in design flood 
def percent_change(value_future, value_historical, true_change): #value_future and value_historical are estimated from a model and true_change is from truth model
    estimated_change_model = ((value_future - value_historical)/value_historical)*100
    return estimated_change_model - true_change


for scenario in scenario:
    #initialize the dataframe
    df_change = pd.DataFrame() #merge final
    df_change_dist = pd.DataFrame() #merge after each distribution
    #loop through each method and distribution
    for method in ['mle']:
        for distribution in ['gev']:
            df_tyr_flood =pd.DataFrame(columns=['model', 'grid', 'comb', '50yr_flood', 'precip_rmse'])
            df_change_flood = pd.DataFrame(columns=['station', 'model', 'change_50yr_flood', 'precip_rmse'])
            # basin_id = '01109060'
            basin_list = pd.read_csv("data/ma29basins.csv", dtype={'basin_id':str})
            used_basin_list = basin_list['basin_id']

            for basin_id in used_basin_list:

                #True precipitation
                #read true precipitation
                true_precip = pd.read_csv(f'data/baseline/true_precip/true_precip{basin_id}.csv')

                #gr4j truth model
                #read the streamflow data
                gr4j_truth = pd.read_csv(f'output/baseline/gr4j_true/gr4j_true{basin_id}.csv')
                gr4j_truth['year'] = gr4j_truth['date'].apply(lambda x: int(x.split('-')[0]))
                data = gr4j_truth.groupby('year')['streamflow'].max()
                
                #calculate the 50 years flood
                flood_50 = return_flood(data,50,distribution,method)
                true_tyr_flood =pd.DataFrame({'model':'Gr4j True', 'grid':'NA', 'comb':'NA', '50yr_flood':flood_50, 'precip_rmse':0}, index=[0])
                df_tyr_flood = pd.concat([df_tyr_flood, true_tyr_flood], ignore_index=True)
                flood_50t = flood_50 #save true floods

                #read the future streamflow data
                gr4j_true_future = pd.read_csv(f'output/{scenario}/gr4j_true/gr4j_true{basin_id}.csv')
                gr4j_true_future['year'] = gr4j_true_future['date'].apply(lambda x: int(x.split('-')[0]))
                data = gr4j_true_future.groupby('year')['streamflow'].max()
                #calculate the 50 years flood
                flood_50 = return_flood(data,50,distribution,method)
                true_tyr_flood_future =pd.DataFrame({'model':'Gr4j True Future', 'grid':'NA', 'comb':'NA', '50yr_flood':flood_50, 'precip_rmse':0}, index=[0])
                df_tyr_flood = pd.concat([df_tyr_flood, true_tyr_flood_future], ignore_index=True)

                #true change in 50yr flood
                true_change_50yr = ((flood_50 - flood_50t)/flood_50t)*100

                #loop through each grid coverage and combination
                grid_list = np.arange(30)
                grid_list = np.append(grid_list, [99])
                for grid in grid_list:
                    for comb in range(15):
                        #--HISTORICAL DATA--#
                        #Interpolated precipitation
                        file_path = f'output/baseline/regr4j/regr4j{basin_id}_coverage{grid}_comb{comb}.csv'
                        if os.path.exists(file_path):
                            idw_precip = pd.read_csv(f'data/baseline/noisy_precip/noisy_precip{basin_id}_coverage{grid}_comb{comb}.csv')
                            precip_rmse = np.sqrt(np.mean((true_precip['PRECIP'] - idw_precip['PRECIP'])**2)) #calculate the rmse
                            if np.array_equal(true_precip['PRECIP'], idw_precip['PRECIP']):
                                rmse_precip = 0
                    
                            #Gr4j Recalibrated model
                            #read the streamflow data
                            gr4j_recalibrated = pd.read_csv(f'output/baseline/regr4j/regr4j{basin_id}_coverage{grid}_comb{comb}.csv')
                            # gr4j_recalibrated['year'] = pd.to_datetime(gr4j_recalibrated['date']).dt.year
                            gr4j_recalibrated['year'] = gr4j_recalibrated['date'].apply(lambda x: int(x.split('-')[0]))
                            data = gr4j_recalibrated.groupby('year')['streamflow'].max()
                            #calculate the 50 years flood
                            flood_50 = return_flood(data,50,distribution,method)
                            temp_df_gr4jr = pd.DataFrame({'model':'Gr4j Recalib', 'grid':grid, 'comb':comb, '50yr_flood':flood_50, 'precip_rmse':precip_rmse}, index=[0])
                            df_tyr_flood = pd.concat([df_tyr_flood, temp_df_gr4jr], ignore_index=True)

                            #Hymod model
                            #read the streamflow data
                            hymod = pd.read_csv(f'output/baseline/simp_hymod/simp_hymod{basin_id}_coverage{grid}_comb{comb}.csv')
                            # hymod['year'] = pd.to_datetime(hymod['date']).dt.year
                            hymod['year'] = hymod['date'].apply(lambda x: int(x.split('-')[0]))
                            data = hymod.groupby('year')['streamflow'].max()
                            #calculate the 50 years flood
                            flood_50 = return_flood(data,50,distribution,method)
                            temp_df_hy = pd.DataFrame({'model':'Hymod', 'grid':grid, 'comb':comb, '50yr_flood':flood_50, 'precip_rmse':precip_rmse}, index=[0])
                            df_tyr_flood = pd.concat([df_tyr_flood, temp_df_hy], ignore_index=True)

                            #LSTM model
                            #read the streamflow data
                            if os.path.exists(f'output/baseline/regional_lstm/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv'):
                                lstm = pd.read_csv(f'output/baseline/regional_lstm/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv')
                                # lstm['year'] = pd.to_datetime(lstm['date']).dt.year
                                lstm['year'] = lstm['date'].apply(lambda x: int(x.split('-')[0]))
                                data = lstm.groupby('year')['streamflow'].max()
                                data = data[data>0]#only keep non zero values
                                #calculate the 50 years flood
                                flood_50 = return_flood(data,50,distribution,method)
                            else:
                                flood_50 = np.NAN

                            temp_df_lstm = pd.DataFrame({'model':'LSTM', 'grid':grid, 'comb':comb, '50yr_flood':flood_50, 'precip_rmse':precip_rmse}, index=[0])
                            df_tyr_flood = pd.concat([df_tyr_flood, temp_df_lstm], ignore_index=True)

                            #HYMOD-LSTM model
                            #read the streamflow data
                            if os.path.exists(f'output/baseline/regional_lstm_simp_hymod/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv'):
                                lstm = pd.read_csv(f'output/baseline/regional_lstm_simp_hymod/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv')
                                # lstm['year'] = pd.to_datetime(lstm['date']).dt.year
                                lstm['year'] = lstm['date'].apply(lambda x: int(x.split('-')[0]))
                                data = lstm.groupby('year')['sim_streamflow'].max()
                                data = data[data>0]#only keep non zero values
                                #calculate the 50 years flood
                                flood_50 = return_flood(data,50,distribution,method)
                            else:
                                flood_50 = np.NAN

                            temp_df_lstm_hymod = pd.DataFrame({'model':'HYMOD-LSTM', 'grid':grid, 'comb':comb, '50yr_flood':flood_50, 'precip_rmse':precip_rmse}, index=[0])
                            df_tyr_flood = pd.concat([df_tyr_flood, temp_df_lstm_hymod], ignore_index=True)

    ##############################################################################################################################################################################################################################################
                            #--FUTURE DATA--#
                            #Gr4j Recalibrated model future
                            #read the streamflow data
                            gr4j_recalibrated_future =pd.read_csv(f'output/{scenario}/regr4j/regr4j{basin_id}_coverage{grid}_comb{comb}.csv')
                            gr4j_recalibrated_future['year'] = gr4j_recalibrated_future['date'].apply(lambda x: int(x.split('-')[0]))
                            data = gr4j_recalibrated_future.groupby('year')['streamflow'].max()
                            #calculate the 50 years flood
                            flood_50 = return_flood(data,50,distribution,method)
                            temp_df_gr4jrf = pd.DataFrame({'model':'Gr4j Recalib Future', 'grid':grid, 'comb':comb, '50yr_flood':flood_50, 'precip_rmse':precip_rmse}, index=[0])
                            df_tyr_flood = pd.concat([df_tyr_flood, temp_df_gr4jrf], ignore_index=True)

                            #Hymod model future
                            #read the streamflow data
                            hymod_future = pd.read_csv(f'output/{scenario}/simp_hymod/simp_hymod{basin_id}_coverage{grid}_comb{comb}.csv')
                            # hymod_future['year'] = pd.to_datetime(hymod_future['date']).dt.year
                            hymod_future['year'] = hymod_future['date'].apply(lambda x: int(x.split('-')[0]))
                            data = hymod_future.groupby('year')['streamflow'].max()
                            #calculate the 50 years flood
                            flood_50 = return_flood(data,50,distribution,method)
                            temp_df_hyf = pd.DataFrame({'model':'Hymod Future', 'grid':grid, 'comb':comb, '50yr_flood':flood_50, 'precip_rmse':precip_rmse}, index=[0])
                            df_tyr_flood = pd.concat([df_tyr_flood, temp_df_hyf], ignore_index=True)

                            #LSTM model future
                            #read the streamflow data
                            if os.path.exists(f'output/{scenario}/regional_lstm/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv'):
                                lstm_future = pd.read_csv(f'output/{scenario}/regional_lstm/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv')
                                # lstm_future['year'] = pd.to_datetime(lstm_future['date']).dt.year
                                lstm_future['year'] = lstm_future['date'].apply(lambda x: int(x.split('-')[0]))
                                data = lstm_future.groupby('year')['streamflow'].max()
                                data = data[data>0]#only keep non zero values
                                #calculate the 50 years flood
                                flood_50 = return_flood(data,50,distribution,method)
                            else:
                                flood_50 = np.NaN
                            temp_df_lstmf = pd.DataFrame({'model':'LSTM Future', 'grid':grid, 'comb':comb, '50yr_flood':flood_50, 'precip_rmse':precip_rmse}, index=[0])
                            df_tyr_flood = pd.concat([df_tyr_flood, temp_df_lstmf], ignore_index=True)

                            #HYMOD-LSTM model future
                            #read the streamflow data
                            if os.path.exists(f'output/{scenario}/regional_lstm_simp_hymod/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv'):
                                lstm_future = pd.read_csv(f'output/{scenario}/regional_lstm_simp_hymod/lstm_input{basin_id}_coverage{grid}_comb{comb}.csv')
                                # lstm_future['year'] = pd.to_datetime(lstm_future['date']).dt.year
                                lstm_future['year'] = lstm_future['date'].apply(lambda x: int(x.split('-')[0]))
                                data = lstm_future.groupby('year')['sim_streamflow'].max()
                                data = data[data>0]#only keep non zero values
                                #calculate the 50 years flood
                                flood_50 = return_flood(data,50,distribution,method)
                            else:
                                flood_50 = np.NaN
                            temp_df_lstm_hymodf = pd.DataFrame({'model':'HYMOD-LSTM Future', 'grid':grid, 'comb':comb, '50yr_flood':flood_50, 'precip_rmse':precip_rmse}, index=[0])
                            df_tyr_flood = pd.concat([df_tyr_flood, temp_df_lstm_hymodf], ignore_index=True)


                            change_gr4j_recalib = pd.DataFrame({'station':basin_id,'model':'Gr4j Recalib',
                                            'change_50yr_flood':percent_change(temp_df_gr4jrf['50yr_flood'] , temp_df_gr4jr['50yr_flood'], true_change_50yr), 'precip_rmse':precip_rmse}, index=[0])

                            change_lstm = pd.DataFrame({'station':basin_id,'model':'LSTM',
                                            'change_50yr_flood':percent_change(temp_df_lstmf['50yr_flood'] , temp_df_lstm['50yr_flood'], true_change_50yr), 'precip_rmse':precip_rmse}, index=[0])

                            change_hymod_lstm = pd.DataFrame({'station':basin_id,'model':'HYMOD-LSTM',
                                            'change_50yr_flood':percent_change(temp_df_lstm_hymodf['50yr_flood'] , temp_df_lstm_hymod['50yr_flood'], true_change_50yr), 'precip_rmse':precip_rmse}, index=[0])

                            change_hymod = pd.DataFrame({'station':basin_id,'model':'Hymod',
                                            'change_50yr_flood':percent_change(temp_df_hyf['50yr_flood'] , temp_df_hy['50yr_flood'], true_change_50yr), 'precip_rmse':precip_rmse}, index=[0])
                            
                            
                            df_change_flood = pd.concat([df_change_flood, change_gr4j_recalib, change_hymod, change_lstm, change_hymod_lstm], ignore_index=True)
                            #add method and distribution to the dataframe
                            df_change_flood['method'] = method
                            df_change_flood['distribution'] = distribution
            #merge the dataframes after each distribution
            df_change_dist = pd.concat([df_change_dist, df_change_flood], ignore_index=True)
        #merge the dataframes after each method and distribution
        df_change = pd.concat([df_change, df_change_dist], ignore_index=True)
    #save the dataframes after all methods and distributions
    df_change.to_csv(f'output/flood_change_{scenario}.csv', index=False)