import numpy as np
import pandas as pd
import os
os.chdir('Z:/MA-Precip-Uncertainty-Exp-Gr4j-Truth')
#read all basin lists
basin_list = pd.read_csv('data/ma29basins.csv',  dtype={'basin_id':str})

#function to calculate rmse value
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value


######---for training dataset---######
#--FOR TRUE PRECIP--#
for id in basin_list['basin_id']:
    lstm_df = pd.read_csv(f'data/baseline/regional_lstm/processed_lstm_train_datasets/pb0/lstm_input{id}.csv')
    gr4j_true = pd.read_csv(f'output/baseline/gr4j_true/gr4j_true{id}.csv')
    simp_hymod = pd.read_csv(f'output/baseline/simp_hymod/simp_hymod{id}_coverage99_comb0.csv')

    #change lstm_df qobs to (gr4j true - simp hymod)
    lstm_df['qobs'] = gr4j_true['streamflow'] - simp_hymod['streamflow']

    #save the final lstm input file
    output_dir = f'data/baseline/regional_lstm_simp_hymod/processed_lstm_train_datasets/pb0'
    output_dir_pred = f'data/baseline/regional_lstm_simp_hymod/processed_lstm_prediction_datasets/pb0'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_pred, exist_ok=True)
    lstm_df.to_csv(f'{output_dir}/lstm_input{id}.csv', index=False)
    lstm_df.to_csv(f'{output_dir_pred}/lstm_input{id}_coverage99_comb0.csv', index=False)


#--HISTORICAL--#
for id in basin_list['basin_id']:
    #read interpolate precip for this basin
    for precip_bucket in ['0-1','1-2','2-3', '3-4', '4-6', '6-8','8-10']:
        random_precip_rmse = 1000
        for coverage in range(15):
            for comb in range(15):
                file_path = f'data/baseline/regional_lstm/processed_lstm_prediction_datasets/pb{precip_bucket}/lstm_input{id}_coverage{coverage}_comb{comb}.csv'
                if os.path.exists(file_path):
                    #true precip
                    true_precip = pd.read_csv(f'data/baseline/true_precip/true_precip{id}.csv')
                    #read interpolated precip
                    lstm_df =pd.read_csv(file_path)
                    precip_rmse = rmse(lstm_df['noisy_precip'], true_precip['PRECIP'])
                    precip_rmse = round(precip_rmse, 3)
                    if precip_rmse < random_precip_rmse:
                        random_precip_rmse = precip_rmse
                        #read true discharge for this basin
                        true_flow = pd.read_csv(f'output/baseline/gr4j_true/gr4j_true{id}.csv')
                        #read hymod discharge
                        hymod_flow = pd.read_csv(f'output/baseline/simp_hymod/simp_hymod{id}_coverage{coverage}_comb{comb}.csv')
                        
                        # change qobs to (qobs from gr4j true - qobs from simp_hymod)
                        lstm_df['qobs'] = true_flow['streamflow'] - hymod_flow['streamflow']
                        # Ensure output directories exist
                        output_dir_pred = f'data/baseline/regional_lstm_simp_hymod/processed_lstm_prediction_datasets/pb{precip_bucket}'
                        output_dir_train = f'data/baseline/regional_lstm_simp_hymod/processed_lstm_train_datasets/pb{precip_bucket}'
                        os.makedirs(output_dir_pred, exist_ok=True)
                        os.makedirs(output_dir_train, exist_ok=True)
                        # Save all files
                        lstm_df.to_csv(f'{output_dir_pred}/lstm_input{id}_coverage{coverage}_comb{comb}.csv', index=False)
                        # Save the final lstm input file
                        lstm_df.to_csv(f'{output_dir_train}/lstm_input{id}.csv', index=False)






for id in basin_list['basin_id']:
    #read interpolate precip for this basin
    for precip_bucket in ['0-1','1-2','2-3', '3-4', '4-6', '6-8','8-10']:
        random_precip_rmse = 1000
        for coverage in range(15):
            for comb in range(15):
                file_path = f'data/baseline/regional_lstm/processed_lstm_prediction_datasets/pb{precip_bucket}/lstm_input{id}_coverage{coverage}_comb{comb}.csv'
                if os.path.exists(file_path):
                    #true precip
                    true_precip = pd.read_csv(f'data/baseline/true_precip/true_precip{id}.csv')
                    #read interpolated precip
                    lstm_df =pd.read_csv(file_path)
                    precip_rmse = rmse(lstm_df['noisy_precip'], true_precip['PRECIP'])
                    precip_rmse = round(precip_rmse, 3)
                    # if precip_rmse < random_precip_rmse:

                    #read true discharge for this basin
                    true_flow = pd.read_csv(f'output/baseline/gr4j_true/gr4j_true{id}.csv')
                    #read hymod discharge
                    hymod_flow = pd.read_csv(f'output/baseline/simp_hymod/simp_hymod{id}_coverage{coverage}_comb{comb}.csv')
                    
                    # change qobs to (qobs from gr4j true - qobs from simp_hymod)
                    lstm_df['qobs'] = true_flow['streamflow'] - hymod_flow['streamflow']
                    # Ensure output directories exist
                    output_dir_pred = f'data/baseline/regional_lstm_simp_hymod/processed_lstm_prediction_datasets/pb{precip_bucket}'
                    output_dir_train = f'data/baseline/regional_lstm_simp_hymod/processed_lstm_train_datasets/pb{precip_bucket}'
                    os.makedirs(output_dir_pred, exist_ok=True)
                    os.makedirs(output_dir_train, exist_ok=True)
                    # Save all files
                    lstm_df.to_csv(f'{output_dir_pred}/lstm_input{id}_coverage{coverage}_comb{comb}.csv', index=False)
                    # Save the final lstm input file but only if precip rmse < random_precip_rmse
                    if precip_rmse < random_precip_rmse:
                        random_precip_rmse = precip_rmse
                        lstm_df.to_csv(f'{output_dir_train}/lstm_input{id}.csv', index=False)