import numpy as np
import pandas as pd
import os

#read all basin lists
basin_list = pd.read_csv('data/ma29basins.csv',  dtype={'basin_id':str})

#function to calculate rmse value
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value

#--FOR TRUE PRECIP--#
for id in basin_list['basin_id']:
    for case in ['', 'future_']: #iterate over historical and future case
        precip_true = pd.read_csv(f'data/regional_lstm/{case}noisy_precip_buckets/pb0/{case}true_precip{id}.csv')
        #extract static features from previous input datasets
        previous_file = pd.read_csv(f'data/regional_lstm/lstm_static_input/lstm_input_{id}.csv').iloc[[0]]
        previous_file = previous_file.loc[:, ['DRAIN_SQKM', 'ELEV_MEAN_M_BASIN', 'ELEV_MAX_M_BASIN', 'ELEV_MIN_M_BASIN', 'ELEV_SITE_M', 'SLOPE_PCT', 'ASPECT_DEGREES', 'ASPECT_NORTHNESS', 'ASPECT_EASTNESS',
                                              'CLAYAVE', 'SILTAVE', 'SANDAVE', 'KFACT_UP', 'RFACT', 'HGA', 'HGB', 'HGC', 'HGD', 'PERMAVE',
                                              'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06', 'MIXEDFORNLCD06', 'SHRUBNLCD06', 'GRASSNLCD06', 'PASTURENLCD06', 'CROPSNLCD06']]
        previous_file = round(previous_file, 4)
        previous_file = pd.concat([previous_file]*len(precip_true))
        previous_file = previous_file.reset_index(drop=True)
        
        #read true discharge for this basin
        true_flow = pd.read_csv(f'output/hbv_true/hbv_true{id}.csv')

        #merge data,temp,flow from true file to previous file
        previous_file['noisy_precip'] = precip_true['PRECIP']
        previous_file['era5temp'] = true_flow['era5temp']
        previous_file['date'] = true_flow['date']
        previous_file['qobs'] = true_flow['streamflow']
        previous_file = previous_file.reset_index(drop=True)

        #save the final lstm input file
        if case == '':
            tag = 'historical'
        else:
            tag = 'future'
        previous_file.to_csv(f'data/regional_lstm/processed_lstm_prediction_datasets/{tag}/pb0/lstm_input{id}_coverage99_comb0.csv', index=False)



#--HISTORICAL--#
for id in basin_list['basin_id']:
    #read interpolate precip for this basin
    for precip_bucket in ['0-1','1-2','2-3', '3-4', '4-6', '6-8','8-10']:
        for coverage in range(15):
            for comb in range(15):
                file_path = f'data/regional_lstm/noisy_precip_buckets/pb{precip_bucket}/noisy_precip{id}_coverage{coverage}_comb{comb}.csv'
                if os.path.exists(file_path):
                    #true precip
                    true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')
                    #read interpolated precip
                    noisy_precip =pd.read_csv(file_path)
                    precip_rmse = rmse(noisy_precip['PRECIP'], true_precip['PRECIP'])
                    precip_rmse = round(precip_rmse, 3)
                    #extract static features from previous input datasets
                    previous_file = pd.read_csv(f'data/regional_lstm/lstm_static_input/lstm_input_{id}.csv').iloc[[0]]
                    previous_file = previous_file.loc[:, ['DRAIN_SQKM', 'ELEV_MEAN_M_BASIN', 'ELEV_MAX_M_BASIN', 'ELEV_MIN_M_BASIN', 'ELEV_SITE_M', 'SLOPE_PCT', 'ASPECT_DEGREES', 'ASPECT_NORTHNESS', 'ASPECT_EASTNESS',
                                            'CLAYAVE', 'SILTAVE', 'SANDAVE', 'KFACT_UP', 'RFACT', 'HGA', 'HGB', 'HGC', 'HGD', 'PERMAVE',
                                            'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06', 'MIXEDFORNLCD06', 'SHRUBNLCD06', 'GRASSNLCD06', 'PASTURENLCD06', 'CROPSNLCD06']]
                    previous_file = round(previous_file, 2)
                    previous_file = pd.concat([previous_file]*len(noisy_precip))
                    previous_file = previous_file.reset_index(drop=True)
                    
                    #read true discharge for this basin
                    true_flow = pd.read_csv(f'output/hbv_true/hbv_true{id}.csv')

                    #merge data,temp,flow from true file to previous file
                    previous_file['noisy_precip'] = noisy_precip['PRECIP']
                    previous_file['era5temp'] = true_flow['era5temp']
                    previous_file['date'] = true_flow['date']
                    previous_file['qobs'] = true_flow['streamflow']
                    previous_file = previous_file.reset_index(drop=True)

                    #save the final lstm input file
                    previous_file.to_csv(f'data/regional_lstm/processed_lstm_prediction_datasets/historical/pb{precip_bucket}/lstm_input{id}_coverage{coverage}_comb{comb}.csv', index=False)



#--FUTURE--#
for id in basin_list['basin_id']:
    #read interpolate precip for this basin
    for precip_bucket in ['0-1','1-2','2-3', '3-4', '4-6', '6-8','8-10']:
        for coverage in range(15):
            for comb in range(15):
                file_path = f'data/regional_lstm/future_noisy_precip_buckets/pb{precip_bucket}/noisy_precip{id}_coverage{coverage}_comb{comb}.csv'
                if os.path.exists(file_path):
                    #true precip
                    true_precip = pd.read_csv(f'data/future/future_true_precip/future_true_precip{id}.csv')
                    #read interpolated precip
                    noisy_precip =pd.read_csv(file_path)
                    precip_rmse = rmse(noisy_precip['PRECIP'], true_precip['PRECIP'])
                    precip_rmse = round(precip_rmse, 3)
                    #extract static features from previous input datasets
                    previous_file = pd.read_csv(f'data/regional_lstm/lstm_static_input/lstm_input_{id}.csv').iloc[[0]]
                    previous_file = previous_file.loc[:, ['DRAIN_SQKM', 'ELEV_MEAN_M_BASIN', 'ELEV_MAX_M_BASIN', 'ELEV_MIN_M_BASIN', 'ELEV_SITE_M', 'SLOPE_PCT', 'ASPECT_DEGREES', 'ASPECT_NORTHNESS', 'ASPECT_EASTNESS',
                                            'CLAYAVE', 'SILTAVE', 'SANDAVE', 'KFACT_UP', 'RFACT', 'HGA', 'HGB', 'HGC', 'HGD', 'PERMAVE',
                                            'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06', 'MIXEDFORNLCD06', 'SHRUBNLCD06', 'GRASSNLCD06', 'PASTURENLCD06', 'CROPSNLCD06']]
                    previous_file = round(previous_file, 2)
                    previous_file = pd.concat([previous_file]*len(noisy_precip))
                    previous_file = previous_file.reset_index(drop=True)
                    
                    #read true discharge for this basin
                    true_flow = pd.read_csv(f'output/hbv_true/hbv_true{id}.csv')

                    #merge data,temp,flow from true file to previous file
                    previous_file['noisy_precip'] = noisy_precip['PRECIP']
                    previous_file['era5temp'] = true_flow['era5temp']
                    previous_file['date'] = true_flow['date']
                    previous_file['qobs'] = true_flow['streamflow']
                    previous_file = previous_file.reset_index(drop=True)

                    #save the final lstm input file
                    previous_file.to_csv(f'data/regional_lstm/processed_lstm_prediction_datasets/future/pb{precip_bucket}/lstm_input{id}_coverage{coverage}_comb{comb}.csv', index=False)