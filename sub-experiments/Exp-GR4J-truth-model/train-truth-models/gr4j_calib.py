
#load libraries
import numpy as np
import pandas as pd
import os
from geneticalgorithm import geneticalgorithm as ga
from gr4j_model import gr4j

########### Calibrate hbv model ###########
#Write a function that calibrates hymod model
def calibNSE(station_id, grid, combination):
    #read input csv file
    df = pd.read_csv(f'data/hbv_input_{station_id}.csv')
    ###LET'S CALIBRATE MODEL FOR first 15 years ###
    calib_time = 5479 #15 years is 5479 days
    p = df["precip"][0:calib_time] #precipitation
    date = df["date"][0:calib_time] #date
    temp = df["tavg"][0:calib_time] #temperature
    latitude = df["latitude"][0:calib_time] #latitude
    q_obs = df["qobs"][0:calib_time]  #validation data / observed flow

    ##genetic algorithm for hbv model calibration
    #reference: https://github.com/rmsolgi/geneticalgorithm
    #write a function you want to minimize
    def nse(pars):
        param_dict = {
        'X1': pars[0],
        'X2': pars[1],
        'X3': pars[2],
        'X4': pars[3]
    }
        q_sim = gr4j(p, temp, date, latitude, param_dict)
        #use first 2 years as spinup
        q_sim = q_sim[730:] #remove first 2 years
        q_obs_inner = q_obs[730:] #remove first 2 years
        #calculate nse
        denominator = np.sum((q_obs_inner - (np.mean(q_obs_inner)))**2)
        numerator = np.sum((q_obs_inner - q_sim)**2)
        nse_value = 1 - (numerator/denominator)
        return -nse_value #minimize this (use negative sign if you need to maximize)

    varbound = np.array([[10, 3000],   # X1: production store capacity
                     [-10, 5],     # X2: groundwater exchange
                     [1, 1000],    # X3: routing store capacity
                     [0.5, 10]])    # X4: unit hydrograph time constant


    algorithm_param = {
        'max_num_iteration': 100,  #100            #100 Generations, higher is better, but requires more computational time
        'max_iteration_without_improv': None,   # Stopping criterion for lack of improvement
        'population_size': 500, #500                #2000 Number of parameter-sets in a single iteration/generation(to start with population 10 times the number of parameters should be fine!)
        'parents_portion': 0.3,                 # Portion of new generation population filled by previous population
        'elit_ratio': 0.01,                     # Portion of the best individuals preserved unchanged
        'crossover_probability': 0.3,           # Chance of existing solution passing its characteristics to new trial solution
        'crossover_type': 'uniform',            # Create offspring by combining the parameters of selected parents
        'mutation_probability': 0.01            # Introduce random changes to the offspring’s parameters (0.1 is 10%)
    }

    model = ga(function = nse,
            dimension = 4, #number of parameters to be calibrated
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
    df_param = df_param.rename(columns={0: 'X1', 1: 'X2', 2: 'X3', 3: 'X4'})
    df_param["station_id"] = str(station_id)
    df_param["nse"] = nse_value
    #save as a csv file
    output_dir = 'output/gr4j_params'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_param.to_csv(f'{output_dir}/gr4j_params_{station_id}.csv', index=False)
    return df_param #returns calibrated parameters and nse value
##End of function

#run model
stations = pd.read_csv('ma29basins.csv', dtype={'basin_id': str})
all_params = []

for id in stations['basin_id']:
    print(f'Calibrating GR4J model for station {id}')
    df = calibNSE(id, 'grid', 'combination')
    print(df)
    all_params.append(df)
    
# Concatenate all parameter DataFrames and save as a single CSV
df_all_params = pd.concat(all_params, ignore_index=True)
df_all_params.to_csv('gr4j_true_params.csv', index=False)

#use saved params to run model and simulate for entire period
for id in stations['basin_id']:
    print(f'Running GR4J model for station {id}')
    #read input csv file
    df = pd.read_csv(f'data/hbv_input_{id}.csv')
    p = df["precip"] #precipitation
    date = df["date"] #date
    temp = df["tavg"] #temperature
    latitude = df["latitude"] #latitude

    #read calibrated parameters
    df_param = pd.read_csv(f'output/gr4j_params/gr4j_params_{id}.csv')
    param_dict = {
        'X1': df_param['X1'][0],
        'X2': df_param['X2'][0],
        'X3': df_param['X3'][0],
        'X4': df_param['X4'][0]
    }
    q_sim = gr4j(p, temp, date, latitude, param_dict)
    #save the output
    output_dir = 'output/gr4j_simulated'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_qsim = pd.DataFrame(q_sim, columns=['sim'])
    df_qsim['sim'] = np.round(df_qsim['sim'], 4) #round to 4 decimal places
    df_qsim['obs'] = df['qobs'].round(4) #round to 4 decimal places
    df_qsim["date"] = date
    df_qsim["station_id"] = str(id)
    df_qsim.to_csv(f'{output_dir}/gr4j_simulated_{id}.csv', index=False)
    print(df_qsim)

print('!!!GR4J model calibration and simulation completed for all stations!!!')