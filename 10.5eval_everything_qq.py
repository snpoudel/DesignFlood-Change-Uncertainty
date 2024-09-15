import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#write a function to calculate RMSE
def rmse(q_obs, q_sim):
    rmse_value = np.sqrt(np.mean((q_obs - q_sim)**2))
    return rmse_value


#read list of basins
basin_list = pd.read_csv('data/MA_basins_gauges_2000-2020_filtered.csv',dtype={'basin_id':str})
#basin grid and combination of interest
id = '01109060'

#precip
true_precip = pd.read_csv(f'data/true_precip/true_precip{id}.csv')
future_true_precip = pd.read_csv(f'data/future/future_true_precip/future_true_precip{id}.csv')
#flow
true_flow = pd.read_csv(f'output/hbv_true_streamflow/hbv_true_output_{id}.csv')
future_true_flow = pd.read_csv(f'output/future/hbv_true_future_streamflow/hbv_true_future_output_{id}.csv')

#function to read precip error and categorize into one of these buckets
def precip_bucket(precip_error):
    if precip_error <=2:
        return('0-2')
    elif precip_error <=4:
        return('2-4')
    elif precip_error <=6:
        return('4-6')
    elif precip_error <=8:
        return('6-8')
    elif precip_error <=10:
        return('8-10')
    else:
        return('>10')
    


df = pd.DataFrame()#dataframe store historical precip data
dff =pd.DataFrame()#dataframe store future precip data

fd = pd.DataFrame()#dataframe store historical flow data
ffd =pd.DataFrame()#dataframe store future flow data

#read interpolated precipitaiton
for coverage in np.arange(30):
    for comb in np.arange(15):
        file_path = f'data/idw_precip/idw_precip{id}_coverage{coverage}_comb{comb}.csv'
        if os.path.exists(file_path):
            idw_precip = pd.read_csv(file_path)
            future_idw_precip=pd.read_csv(f'data/future/future_idw_precip/future_idw_precip{id}_coverage{coverage}_comb{comb}.csv')
            precip_rmse = rmse(true_precip['PRECIP'], idw_precip['PRECIP'])
            pb = precip_bucket(precip_rmse) #categorize into a precip bucket
            idw_precip['tag'] = pb
            idw_precip['truth'] = true_precip['PRECIP']
            future_idw_precip['tag'] = pb
            future_idw_precip['truth'] = future_true_precip['PRECIP']
            df = pd.concat([df, idw_precip])
            dff = pd.concat([dff, future_idw_precip])

            #Read corresponding streamflow for HBV recalibrated model
            hbvre = pd.read_csv(f'output/hbv_idw_recalib_streamflow/hbv_idw_recalib_streamflow{id}_coverage{coverage}_comb{comb}.csv')
            future_hbvre = pd.read_csv(f'output/future/hbv_idw_recalib_future_streamflow/hbv_idw_recalib_future_streamflow{id}_coverage{coverage}_comb{comb}.csv')
            hbvre['tag'] = pb
            hbvre['truth'] = true_flow['streamflow']
            hbvre['model'] = 'HBV Re'
            future_hbvre['tag'] =pb
            future_hbvre['truth'] = future_true_flow['streamflow']
            future_hbvre['model'] = 'HBV Re'
            fd = pd.concat([fd, hbvre])
            ffd  = pd.concat([ffd, future_hbvre])

            #Read corresponding streamflow for hymod  model
            hymod = pd.read_csv(f'output/hymod_idw_streamflow/hymod_interpol_streamflow{id}_coverage{coverage}_comb{comb}.csv')
            future_hymod = pd.read_csv(f'output/future/hymod_idw_future_streamflow/hymod_interpol_future_streamflow{id}_coverage{coverage}_comb{comb}.csv')
            hymod['tag'] = pb
            hymod['truth'] = true_flow['streamflow']
            hymod['model'] = 'HYMOD'
            future_hymod['tag'] =pb
            future_hymod['truth'] = future_true_flow['streamflow']
            future_hymod['model'] = 'HYMOD'
            fd = pd.concat([fd, hymod])
            ffd  = pd.concat([ffd, future_hymod])

            #Read corresponding streamflow for lstm  model
            lstm = pd.read_csv(f'output/lstm_idw_streamflow/lstm_idw_{id}_coverage{coverage}_comb{comb}.csv')
            future_lstm = pd.read_csv(f'output/future/lstm_idw_future_streamflow/lstm_idw_future_streamflow{id}_coverage{coverage}_comb{comb}.csv')
            lstm['tag'] = pb
            true_flow_forlstm = true_flow[365:].reset_index(drop=True)
            lstm['truth'] = true_flow_forlstm['streamflow']
            lstm['model'] = 'LSTM'
            future_lstm['tag'] =pb
            future_true_flow_forlstm = future_true_flow[365:].reset_index(drop=True)
            future_lstm['truth'] = future_true_flow_forlstm['streamflow']
            future_lstm['model'] = 'LSTM'
            fd = pd.concat([fd, lstm])
            ffd  = pd.concat([ffd, future_lstm])


#save the plots
df.to_csv(f'output/hist_precip_by_buckets_{id}.csv', index=False)
dff.to_csv(f'output/future_precip_by_buckets_{id}.csv', index=False)
fd.to_csv(f'output/hist_flow_by_buckets_{id}.csv', index=False)
ffd.to_csv(f'output/future_flow_by_buckets_{id}.csv', index=False)
