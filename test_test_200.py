#load libraries
import numpy as np
import pandas as pd
import os
from geneticalgorithm import geneticalgorithm as ga
from hbv_model import hbv
# from mpi4py import MPI

########### Calibrate hbv model ###########
#Write a function that calibrates hymod model
def calibNSE(station_id, grid, combination):
    #read input csv file
    #read interpolated precipitation
    df = pd.read_csv(f'data/idw_precip/idw_precip{station_id}_coverage{grid}_comb{combination}.csv')
    ###LET'S CALIBRATE MODEL FOR 50 % data ###
    calib_time = 5115 #calibrate for 5115 days, about 14 years
    p = df["PRECIP"][0:calib_time] #precipitation
    date = df["DATE"][0:calib_time] #date

    #read temperature, latitude, and observed flow (true hbv flow)
    df_true = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{station_id}.csv')
    temp = df_true["era5temp"][0:calib_time] #temperature
    latitude = df_true["latitude"][0:calib_time] #latitude
    routing = 1 # 0: no routing, 1 allows running
    q_obs = df_true["streamflow"][0:calib_time]  #validation data / observed flow

    ##genetic algorithm for hbv model calibration
    #reference: https://github.com/rmsolgi/geneticalgorithm
    #write a function you want to minimize
    def nse(pars):
        q_sim = hbv(pars, p, temp, date, latitude, routing)
        denominator = np.sum((q_obs - (np.mean(q_obs)))**2)
        numerator = np.sum((q_obs - q_sim)**2)
        nse_value = 1 - (numerator/denominator)
        return -nse_value #minimize this (use negative sign if you need to maximize)

    varbound = np.array([[1,1000], #fc
                        [1,7], #beta
                        [0.01,0.99], #pwp
                        [1,999], #l
                        [0.01,0.99], #ks
                        [0.01,0.99], #ki
                        [0.0001, 0.99], #kb
                        [0.001, 0.99], #kperc
                        [0.5,2], #coeff_pet
                        [0.01,10], #ddf
                        [0.5,1.5], #scf
                        [-1,4], #ts
                        [-1,4], #tm
                        [-1,4], #tti
                        [0, 0.2], #whc
                        [0.1,1], #crf
                        [1,10]]) #maxbas

    algorithm_param = {
        'max_num_iteration': 100,              #100 Generations, higher is better, but requires more computational time
        'max_iteration_without_improv': None,   # Stopping criterion for lack of improvement
        'population_size': 200,                 #200 Number of parameter-sets in a single iteration/generation(to start with population 10 times the number of parameters should be fine!)
        'parents_portion': 0.3,                 # Portion of new generation population filled by previous population
        'elit_ratio': 0.01,                     # Portion of the best individuals preserved unchanged
        'crossover_probability': 0.3,           # Chance of existing solution passing its characteristics to new trial solution
        'crossover_type': 'uniform',            # Create offspring by combining the parameters of selected parents
        'mutation_probability': 0.01            # Introduce random changes to the offspring’s parameters (0.1 is 10%)
    }

    model = ga(function = nse,
            dimension = 17, #number of parameters to be calibrated
            variable_type= 'real',
            variable_boundaries = varbound,
            algorithm_parameters = algorithm_param)

    model.run()
    #end of genetic algorithm

    #output of the genetic algorithm/best parameters
    best_parameters = model.output_dict
    param_value = best_parameters["variable"]
    nse_value = best_parameters["function"]
    nse_value = -nse_value #nse function gives -ve values, which is now reversed here to get true nse
    #convert into a dataframe
    df_param = pd.DataFrame(param_value).transpose()
    df_param = df_param.rename(columns={0:"fc", 1:"beta", 2:"pwp", 3:"l", 4:"ks", 5:"ki",
                             6:"kb", 7:"kperc",  8:"coeff_pet", 9:"ddf", 10:"scf", 11:"ts",
                             12:"tm", 13:"tti", 14:"whc", 15:"crf", 16:"maxbas"})
    df_param["station_id"] = str(station_id)
    df_param["nse"] = nse_value
    #save as a csv file
    return df_param #returns calibrated parameters and nse value
##End of function


#read true parameters
true_params = pd.read_csv('data/true_hbv_calibrated_parameters.csv', dtype={'station_id':str})
true_params = true_params[true_params['station_id']==id]
true_params = true_params.iloc[:, :-3]
total_df = pd.DataFrame(true_params).reset_index(drop=True)
for iter in range(5):
    #basin id
    #id = '01108000'
    lat_basin = pd.read_csv('data/basinID_withLatLon.csv', dtype={'STAID':str})

    basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv', dtype={'basin_id':str})
    used_basin_list = ['01170500', '01108000', '01104500', '01109060', '01177000']
    used_basin_list_grid = [used_basin_list[0]]*30 + [used_basin_list[1]]*11 + [used_basin_list[2]]*7 + [used_basin_list[3]]*7+ [used_basin_list[4]]*6
    #length of used basin list grid is 61
    id = used_basin_list_grid[0]
    grid = 99
    combination = 0

    #---HISTORICAL OBSERVATION---#
    precip_in = pd.read_csv(f'data/idw_precip/idw_precip{id}_coverage{grid}_comb{combination}.csv')
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
    params_in = calibNSE(id, grid, combination)
    calib_params = params_in
    #calib_params = calib_params.iloc[:,:-2]

    #combine dataframes
    total_df = pd.concat([total_df, calib_params], ignore_index=True)
    print(total_df)
total_df.to_csv('output/param_test_plots/pop200_gen100.csv', index=False)