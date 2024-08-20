#load libraries
import numpy as np
import pandas as pd
import os
from hbv_model import hbv
import multiprocessing

#basin id
id = '01108000'
#only keep stations that has hbv parameters
hbv_calib_parameters = pd.read_csv('data/basin_id_withshapefile_new.csv', dtype=str)

#read the list of basin ID with centriod latitude
lat_basin = pd.read_csv('data/basinID_withLatLon.csv', dtype={'STAID':str})
#read calibrated hbv parameters
hbv_params = pd.read_csv('data/true_hbv_calibrated_parameters.csv', dtype = {'station_id':str})
hbv_params = hbv_params.iloc[:,:-2] #remove undesired columns

#make the things to be parallel processed into a function
def func_multiprocess(grid):
    for combination in range(10):
        #Read interpolated precipitation
        file_location = f'data/idw_precip/idw_precip{id}_coverage{grid}_comb{combination}.csv'
        if os.path.exists(file_location):
            precip_in = pd.read_csv(file_location)
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
            output_df = pd.DataFrame({ 'date':precip_in['DATE'], 'streamflow':q_sim })
            #save output dataframe
            output_df.to_csv(f'output/hbv_idw_streamflow/hbv_idw_streamflow{id}_coverage{grid}_comb{combination}.csv')
    #End of inner Loop


# Create multiprocessing pool
if __name__ == '__main__': #this ensures that the code is being run in the main module and this block is not run to avoid creating new processes recursively
    pool = multiprocessing.Pool(processes=10)
    grid_list = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    pool.map(func_multiprocess, grid_list)
    pool.close()
    pool.join()
