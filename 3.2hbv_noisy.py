#load libraries
import numpy as np
import pandas as pd
import os
from hbv_model import hbv
import multiprocessing

#basin id
#id = '01108000'
basin_list = pd.read_csv('data/ma29basins.csv', dtype={'basin_id':str})
# used_basin_list = ['01108000', '01104500', '01109060', '01177000']
used_basin_list = basin_list['basin_id']
def func_multiprocess(id):
    #read the list of basin ID with centriod latitude
    lat_basin = pd.read_csv('data/basinID_withLatLon.csv', dtype={'STAID':str})
    #read calibrated hbv parameters
    hbv_params = pd.read_csv('data/true_hbv_calibrated_parameters.csv', dtype = {'station_id':str})
    hbv_params = hbv_params.iloc[:,:-2] #remove undesired columns
    for coverage in range(101):
        for combination in range(15):
            file_path =f'data/noisy_precip/noisy_precip{id}_coverage{coverage}_comb{combination}.csv'
            if os.path.exists(file_path):
                #---HISTORICAL OBSERVATION---#
                precip_in = pd.read_csv(f'data/noisy_precip/noisy_precip{id}_coverage{coverage}_comb{combination}.csv')
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
                output_df = pd.DataFrame({ 'date':precip_in['DATE'], 'streamflow':q_sim })
                #save output dataframe
                output_df.to_csv(f'output/hbv_noisy/hbv_noisy{id}_coverage{coverage}_comb{combination}.csv')

                #---FUTURE OBSERVATION---#
                precip_in = pd.read_csv(f'data/future/future_noisy_precip/future_noisy_precip{id}_coverage{coverage}_comb{combination}.csv')
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
                output_df = pd.DataFrame({ 'date':precip_in['DATE'], 'streamflow':q_sim })
                #save output dataframe
                output_df.to_csv(f'output/future/hbv_noisy/hbv_noisy{id}_coverage{coverage}_comb{combination}.csv')


# Create multiprocessing pool
if __name__ == '__main__': #this ensures that the code is being run in the main module and this block is not run to avoid creating new processes recursively
    pool = multiprocessing.Pool(processes=10)
    grid_list = used_basin_list
    pool.map(func_multiprocess, grid_list)
    pool.close()
    pool.join()
