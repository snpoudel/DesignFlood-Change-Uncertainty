#load libraries
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from hbv_model import hbv
from mpi4py import MPI

#Set up communicator to parallelize job in cluster using MPI
comm = MPI.COMM_WORLD #Get the default communicator object
rank = comm.Get_rank() #Get the rank of the current process
size = comm.Get_size() #Get the total number of processes

########### Calibrate hbv model ###########
#Write a function that calibrates hymod model
def calibNSE(station_id, grid, combination):
    #read input csv file
    #read interpolated precipitation
    df = pd.read_csv(f'data/idw_precip/idw_precip{station_id}_coverage{grid}_comb{combination}.csv')
    ###LET'S CALIBRATE MODEL FOR 50 % data ###
    calib_time = 5115 #calibrate for 5000 days, about 14 years
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
        'max_num_iteration': 100,              # Generations, higher is better, but requires more computational time
        'max_iteration_without_improv': None,   # Stopping criterion for lack of improvement
        'population_size': 200,                 # Number of parameter-sets in a single iteration/generation(to start with population 10 times the number of parameters should be fine!)
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


#basin id
id = '01108000'
#read the list of basin ID with centriod latitude
lat_basin = pd.read_csv('data/basinID_withLatLon.csv', dtype={'STAID':str})
#generate sets of precipitation dataset with different gridded data coverage and different combinatoin of grids coverage
station_coverage = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
combination_sets = np.arange(10)
combination_sets = np.tile(combination_sets, 10)

grid = station_coverage[rank//10] #select grid coverage based on rank
combination = combination_sets[rank]

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
params_in = params_in.iloc[0,:-2] #remove basin ID column
params_in = np.array(params_in)

#run hbv model
q_sim = hbv(params_in, precip_in['PRECIP'], temp_in['t2m'], precip_in['DATE'], lat_in, routing=1)
q_sim = np.round(q_sim, 4)

#keep result in a dataframe
output_df = pd.DataFrame({ 'date':precip_in['DATE'], 'streamflow':q_sim })
#save output dataframe
output_df.to_csv(f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{id}_coverage{grid}_comb{combination}.csv')