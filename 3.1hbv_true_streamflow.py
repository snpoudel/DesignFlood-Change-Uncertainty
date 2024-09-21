#load libraries
import numpy as np
import pandas as pd
from hbv_model import hbv


#read the list of basin ID
#id = '01108000'
basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv', dtype={'basin_id':str})

for id in basin_list['basin_id']:
    #read the list of basin ID with centriod latitude
    lat_basin = pd.read_csv('data/basinID_withLatLon.csv', dtype={'STAID':str})
    #read calibrated hbv parameters
    hbv_params = pd.read_csv('data/true_hbv_calibrated_parameters.csv', dtype = {'station_id':str})
    hbv_params = hbv_params.iloc[:,:-2] #remove undesired columns


    #---HISTORICAL OBSERVATION---#
    precip_in = pd.read_csv(f'data/true_precip/true_precip{id}.csv') 
    #Read temperature era5
    temp_in = pd.read_csv(f'data/processed-era5-temp/temp_{id}.csv')
    #filter temperature for the year 2000-2020
    temp_in = temp_in[temp_in['time'] >= '2000-01-01']
    temp_in = temp_in[temp_in['time'] <= '2020-12-31']
    temp_in = temp_in.reset_index(drop=True)
    #Read latitude
    lat_in_df = lat_basin[lat_basin['STAID'] == id]
    lat_in = lat_in_df['LAT_CENT'].iloc[0]

    #Read calibrated hbv parameters
    params_in = hbv_params[hbv_params['station_id'] == id] #extract parameter for this basin ID
    params_in = params_in.iloc[0,:-1] #remove basin ID column
    params_in = np.array(params_in)
    #run hbv model
    q_sim = hbv(params_in, precip_in['PRECIP'], temp_in['t2m'], precip_in['DATE'], lat_in, routing=1)
    q_sim = np.round(q_sim, 4)
    #keep result in a dataframe
    output_df = pd.DataFrame({'date':precip_in['DATE'], 'true_precip':precip_in['PRECIP'],'era5temp':temp_in['t2m'],
                            'latitude':lat_in, 'streamflow':q_sim })
    #save output dataframe
    output_df.to_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv', index=False)



    #---FUTURE OBSERVATION---#
    precip_in = pd.read_csv(f'data/future/future_true_precip/future_true_precip{id}.csv') 
    #Read temperature era5
    temp_in = pd.read_csv(f'data/processed-era5-temp/temp_{id}.csv')
    #filter temperature for the year 2000-2020
    temp_in = temp_in[temp_in['time'] >= '2000-01-01']
    temp_in = temp_in[temp_in['time'] <= '2020-12-31']
    temp_in = temp_in.reset_index(drop=True)
    #Read latitude
    lat_in_df = lat_basin[lat_basin['STAID'] == id]
    lat_in = lat_in_df['LAT_CENT'].iloc[0]

    #Read calibrated hbv parameters
    params_in = hbv_params[hbv_params['station_id'] == id] #extract parameter for this basin ID
    params_in = params_in.iloc[0,:-1] #remove basin ID column
    params_in = np.array(params_in)
    #run hbv model
    q_sim = hbv(params_in, precip_in['PRECIP'], temp_in['t2m'], precip_in['DATE'], lat_in, routing=1)
    q_sim = np.round(q_sim, 4)
    #keep result in a dataframe
    output_df = pd.DataFrame({'date':precip_in['DATE'], 'true_precip':precip_in['PRECIP'],'era5temp':temp_in['t2m'],
                            'latitude':lat_in, 'streamflow':q_sim })
    #save output dataframe
    output_df.to_csv(f'output/future/hbv_true_future_streamflow/hbv_true_future_output_{id}.csv', index=False)