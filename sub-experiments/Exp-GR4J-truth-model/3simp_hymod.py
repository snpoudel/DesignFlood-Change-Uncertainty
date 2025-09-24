#load libraries
import numpy as np
import pandas as pd
import os
from simp_hymod_model import hymod
import multiprocessing

#basin id
#id = '01108000'
basin_list = pd.read_csv('data/ma29basins.csv', dtype={'basin_id':str})
used_basin_list = basin_list['basin_id']
scenario_list = ['scenario3', 'scenario7', 'scenario11', 'scenario15']
def func_multiprocess(id):
    #read the list of basin ID with centriod latitude
    lat_basin = pd.read_csv('data/basinID_withLatLon.csv', dtype={'STAID':str})
    for scenario in scenario_list:
        for coverage in range(101):
            for combination in range(15):
                file_path =f'data/scenario3/noisy_precip/future_noisy_precip{id}_coverage{coverage}_comb{combination}.csv'
                if os.path.exists(file_path):
                    precip_in = pd.read_csv(f'data/{scenario}/noisy_precip/future_noisy_precip{id}_coverage{coverage}_comb{combination}.csv')
                    #Read temperature era5
                    temp_in = pd.read_csv(f'data/{scenario}/temperature/future_temp{id}.csv')
                    #Read latitude
                    lat_in_df = lat_basin[lat_basin['STAID'] == id]
                    lat_in = lat_in_df['LAT_CENT'].iloc[0]

                    #read calibrated simp_hymod parameters
                    simp_hymod_params = pd.read_csv(f'output/parameters/simp_hymod/params{id}_grid{coverage}_comb{combination}.csv', dtype = {'station_id':str})
                    # simp_hymod_params = simp_hymod_params.iloc[:,:-2] #remove undesired columns

                    params_in = simp_hymod_params.drop(columns=['Unnamed: 0', 'station_id', 'nse'])
                    params_in = np.array(params_in).flatten()
                    #run simp_hymod model
                    q_sim = hymod(params_in, precip_in['PRECIP'], temp_in['tavg'], precip_in['DATE'], lat_in, routing=1)
                    q_sim = np.round(q_sim, 4)
                    output_df = pd.DataFrame({ 'date':precip_in['DATE'], 'streamflow':q_sim })
                    #save output dataframe
                    output_df.to_csv(f'output/{scenario}/simp_hymod/simp_hymod{id}_coverage{coverage}_comb{combination}.csv')


# Create multiprocessing pool
if __name__ == '__main__': #this ensures that the code is being run in the main module and this block is not run to avoid creating new processes recursively
    pool = multiprocessing.Pool(processes=20)
    grid_list = used_basin_list
    pool.map(func_multiprocess, grid_list)
    pool.close()
    pool.join()
