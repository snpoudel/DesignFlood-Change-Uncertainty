#load libraries
import numpy as np
import pandas as pd
import os
from geneticalgorithm import geneticalgorithm as ga
from hbv_model import hbv
import multiprocessing
# from mpi4py import MPI

########### Calibrate hbv model ###########
#Write a function that calibrates hymod model
def calibNSE(index):
    total_df = pd.DataFrame()
    used_basin_list = ['01170500', '01108000', '01104500', '01109060', '01177000']
    #length of used basin list grid is 61
    station_id = used_basin_list[0]
    grid = 99
    combination = 0
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
        'max_num_iteration': 200,              #100 Generations, higher is better, but requires more computational time
        'max_iteration_without_improv': None,   # Stopping criterion for lack of improvement
        'population_size': 2000,                 #200 Number of parameter-sets in a single iteration/generation(to start with population 10 times the number of parameters should be fine!)
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
    df_param.to_csv(f'output/param_test_plots/pop2000_gen200_{index}.csv', index=False)
    return df_param
##End of function

# Create multiprocessing pool
if __name__ == '__main__': #this ensures that the code is being run in the main module and this block is not run to avoid creating new processes recursively
    pool = multiprocessing.Pool(processes=5)
    pool.map(calibNSE, range(5))
    pool.close()
    pool.join()


# #read true parameters
# used_basin_list = ['01170500', '01108000', '01104500', '01109060', '01177000']
# #length of used basin list grid is 61
# id = used_basin_list[0]
# total_df = pd.DataFrame()
# true_params = pd.read_csv('data/true_hbv_calibrated_parameters.csv', dtype={'station_id':str})
# true_params = true_params[true_params['station_id']==id]
# true_params = true_params.iloc[:, :-3]
# file1 = pd.read_csv('output/param_test_plots/pop2000_gen200_0.csv')
# file2 = pd.read_csv('output/param_test_plots/pop2000_gen200_1.csv')
# file3 = pd.read_csv('output/param_test_plots/pop2000_gen200_2.csv')
# file4 = pd.read_csv('output/param_test_plots/pop2000_gen200_3.csv')
# file5 = pd.read_csv('output/param_test_plots/pop2000_gen200_4.csv')
# #merge
# total_df = pd.concat([total_df, true_params, file1, file2, file3, file4, file5], ignore_index=True)
# total_df.to_csv('output/param_test_plots/pop2000_gen200.csv', index=False)
# print(total_df)