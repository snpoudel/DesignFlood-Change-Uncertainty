import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#function to calculate rmse value
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value
    
#read all basin lists
basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv', sep='\t', dtype={'basin_id':str})
df_precip_rmse = pd.DataFrame() #empty dataframe to store values
#loop through each basin, read in true and interpol precip and find precip rmse
for id in basin_list['basin_id']:
    #true precip
    true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')
    #loop through different combination of interpolate precip for this basin
    for coverage in np.arange(15): #this should include all possible coverage values
        for comb in np.arange(15): #this should include all possible combination values
            file_path = f'data/idw_precip/idw_precip{id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                interpol_precip = pd.read_csv(file_path)
                precip_rmse = rmse(interpol_precip['PRECIP'], true_precip['PRECIP'])
                precip_rmse = round(precip_rmse, 3)
                temp_df = pd.DataFrame({'station_id':id, 'num_gauge':coverage, 'comb':comb,
                                         'precip_rmse':precip_rmse}, index=[0])
                df_precip_rmse = pd.concat([df_precip_rmse,temp_df], ignore_index=True)

#convert basin into precip category group
df_precip_rmse['precip_bucket'] = pd.cut(df_precip_rmse['precip_rmse'], bins=[0,2,4,6,8,10],
                                          labels=['0-2', '2-4', '4-6', '6-8', '8-10'])

#make a histogram plot of precip_rmse
plt.hist(df_precip_rmse['precip_rmse'], bins=15, edgecolor = 'black')
plt.xlabel('Precipitation RMSE (mm/day)')
plt.ylabel('Number of basins')


#--HISTORICAL--##
#save precipitation datasets into respective precip buckets
for id in basin_list['basin_id']:
    #true precip
    true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')
    #loop through different combination of interpolate precip for this basin
    for coverage in np.arange(15): #this should include all possible coverage values
        for comb in np.arange(15): #this should include all possible combination values
            file_path = f'data/idw_precip/idw_precip{id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                interpol_precip = pd.read_csv(file_path)
                #find precip bucket for this 
                filtered_df = df_precip_rmse[(df_precip_rmse['station_id']==id) & (df_precip_rmse['num_gauge']==coverage) &
                                             (df_precip_rmse['comb']==comb)]
                precip_bucket = filtered_df['precip_bucket']
                precip_bucket = precip_bucket.iloc[0]
                new_path = f'data/regional_lstm/idw_precip_buckets/pb{precip_bucket}/idw_precip{id}_coverage{coverage}_comb{comb}.csv'
                #save csv file
                interpol_precip.to_csv(new_path, index=False)


#--FUTURE--##
#save precipitation datasets into respective precip buckets
for id in basin_list['basin_id']:
    #true precip future
    future_true_precip = pd.read_csv(f'data/future/future_true_precip/future_true_precip{id}.csv')
    #loop through different combination of interpolate precip for this basin
    for coverage in np.arange(15): #this should include all possible coverage values
        for comb in np.arange(15): #this should include all possible combination values
            file_path = f'data/future/future_idw_precip/future_idw_precip{id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                interpol_precip = pd.read_csv(file_path)
                #find precip bucket for this 
                filtered_df = df_precip_rmse[(df_precip_rmse['station_id']==id) & (df_precip_rmse['num_gauge']==coverage) &
                                             (df_precip_rmse['comb']==comb)]
                precip_bucket = filtered_df['precip_bucket']
                precip_bucket = precip_bucket.iloc[0]
                new_path = f'data/regional_lstm/future_idw_precip_buckets/pb{precip_bucket}/idw_precip{id}_coverage{coverage}_comb{comb}.csv'
                #save csv file
                interpol_precip.to_csv(new_path, index=False)