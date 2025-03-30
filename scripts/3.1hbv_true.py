#load libraries
import numpy as np
import pandas as pd
from hbv_model import hbv


#read the list of basin ID
#id = '01108000'
basin_list = pd.read_csv('data/ma29basins.csv', dtype={'basin_id':str})

for id in basin_list['basin_id']:
    #read the list of basin ID with centriod latitude
    lat_basin = pd.read_csv('data/basinID_withLatLon.csv', dtype={'STAID':str})
    #read calibrated hbv parameters
    hbv_params = pd.read_csv('data/true_hbv_calibrated_parameters.csv', dtype = {'station_id':str})
    hbv_params = hbv_params.iloc[:,:-2] #remove undesired columns


    #---HISTORICAL OBSERVATION---#
    precip_in = pd.read_csv(f'data/true_precip/true_precip{id}.csv') 
    #Read temperature era5
    temp_in = pd.read_csv(f'data/temperature/temp{id}.csv')
    #Read latitude
    lat_in_df = lat_basin[lat_basin['STAID'] == id]
    lat_in = lat_in_df['LAT_CENT'].iloc[0]

    #Read calibrated hbv parameters
    params_in = hbv_params[hbv_params['station_id'] == id] #extract parameter for this basin ID
    params_in = params_in.iloc[0,:-1] #remove basin ID column
    params_in = np.array(params_in)
    #run hbv model
    q_sim = hbv(params_in, precip_in['PRECIP'], temp_in['tavg'], precip_in['DATE'], lat_in, routing=1)
    q_sim = np.round(q_sim, 4)
    #keep result in a dataframe
    output_df = pd.DataFrame({'date':precip_in['DATE'], 'true_precip':precip_in['PRECIP'],'era5temp':temp_in['tavg'],
                            'latitude':lat_in, 'streamflow':q_sim })
    #save output dataframe
    output_df.to_csv(f'output/hbv_true/hbv_true{id}.csv', index=False)



    #---FUTURE OBSERVATION---#
    precip_in = pd.read_csv(f'data/future/future_true_precip/future_true_precip{id}.csv') 
    #Read temperature era5
    temp_in = pd.read_csv(f'data/future/future_temperature/future_temp{id}.csv')
    #Read latitude
    lat_in_df = lat_basin[lat_basin['STAID'] == id]
    lat_in = lat_in_df['LAT_CENT'].iloc[0]

    #Read calibrated hbv parameters
    params_in = hbv_params[hbv_params['station_id'] == id] #extract parameter for this basin ID
    params_in = params_in.iloc[0,:-1] #remove basin ID column
    params_in = np.array(params_in)
    #run hbv model
    q_sim = hbv(params_in, precip_in['PRECIP'], temp_in['tavg'], precip_in['DATE'], lat_in, routing=1)
    q_sim = np.round(q_sim, 4)
    #keep result in a dataframe
    output_df = pd.DataFrame({'date':precip_in['DATE'], 'true_precip':precip_in['PRECIP'],'era5temp':temp_in['tavg'],
                            'latitude':lat_in, 'streamflow':q_sim })
    #save output dataframe
    output_df.to_csv(f'output/future/hbv_true/hbv_true{id}.csv', index=False)