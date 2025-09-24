#load libraries
import numpy as np
import pandas as pd
from gr4j_model import gr4j

#read the list of basin ID
basin_list = pd.read_csv('data/ma29basins.csv', dtype={'basin_id':str})

scenario = ['baseline', 'scenario3', 'scenario7', 'scenario11', 'scenario15' ]


for scenario_name in scenario:
    #read the list of basin ID with centriod latitude
    lat_basin = pd.read_csv('data/basinID_withLatLon.csv', dtype={'STAID':str})
    #read calibrated gr4j parameters
    gr4j_params = pd.read_csv(f'data/gr4j_true_params.csv', dtype = {'station_id':str})
    gr4j_params = gr4j_params.iloc[:,:-1] #remove undesired columns

    for id in basin_list['basin_id']:
        #---HISTORICAL OBSERVATION---#
        if scenario_name == 'baseline':
            precip_in = pd.read_csv(f'data/{scenario_name}/true_precip/true_precip{id}.csv')
        else:
            precip_in = pd.read_csv(f'data/{scenario_name}/true_precip/future_true_precip{id}.csv')

        if scenario_name == 'baseline':
            temp_in = pd.read_csv(f'data/{scenario_name}/temperature/temp{id}.csv')
        else:
            temp_in = pd.read_csv(f'data/{scenario_name}/temperature/future_temp{id}.csv')
        #Read latitude
        lat_in_df = lat_basin[lat_basin['STAID'] == id]
        lat_in = lat_in_df['LAT_CENT'].iloc[0]

        #Read calibrated gr4j parameters
        params_in = gr4j_params[gr4j_params['station_id'] == id] #extract parameter for this basin ID
        params_in = params_in.iloc[0,:-1] #remove basin ID column
        params_in = np.array(params_in)
        params_dict = {
            'X1': params_in[0],
            'X2': params_in[1],
            'X3': params_in[2],
            'X4': params_in[3]
        }
        #run gr4j model
        q_sim = gr4j(precip_in['PRECIP'], temp_in['tavg'], precip_in['DATE'], lat_in, params_dict)
        #round the result to 4 decimal places
        q_sim = np.round(q_sim, 4)
        #keep result in a dataframe
        output_df = pd.DataFrame({'date':precip_in['DATE'], 'true_precip':precip_in['PRECIP'],'era5temp':temp_in['tavg'],
                                'latitude':lat_in, 'streamflow':q_sim })
        #save output dataframe
        output_df.to_csv(f'output/{scenario_name}/gr4j_true/gr4j_true{id}.csv', index=False)