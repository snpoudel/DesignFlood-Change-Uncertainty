import numpy as np
import pandas as pd
from datetime import date
import time
import os
import gc
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mpi4py import MPI

# # # #Set up communicator to parallelize job in cluster using MPI
comm = MPI.COMM_WORLD #Get the default communicator object
rank = comm.Get_rank() #Get the rank of the current process
size = comm.Get_size() #Get the total number of processes
# rank = 1
#precip buckets
precip_buckets = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
pb = 'pb' + precip_buckets[rank]

start_time = time.time()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
NUM_INPUT_FEATURES = 29
NUM_OUTPUT_FEATURES = 1
NUM_EPOCHS = 20#20
NUM_HIDDEN_LAYERS = 1
SEQUENCE_LENGTH = 365
NUM_HIDDEN_NEURONS = 256#256
BATCH_SIZE = 64
LEARNING_RATE = 0.0005#0.0001
DROPOUT_RATE = 0.2

# Function to set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# Function to create LSTM input sequences
def create_sequences(features, targets, sequence_length):
    n_days = len(features) - sequence_length + 1
    x_sequences, y_sequences = [], []
    for i in range(n_days):
        x_seq = features[i:i+sequence_length]
        y_seq = targets[i+sequence_length-1]
        x_sequences.append(x_seq)
        y_sequences.append(y_seq)
    return np.array(x_sequences), np.array(y_sequences)

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_neurons, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_neurons, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_neurons, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(NUM_HIDDEN_LAYERS, x.size(0), NUM_HIDDEN_NEURONS).to(device)
        c0 = torch.zeros(NUM_HIDDEN_LAYERS, x.size(0), NUM_HIDDEN_NEURONS).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# Dataset Class
class SeqDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

#### Train the model for training period
set_seed(10)

# Dates and basins for training
# basin_list = ['01095375', '01096000', '01097000', '01103280', '01104500', '01105000']
basin_list = pd.read_csv("data/ma29basins.csv", dtype={'basin_id':str})['basin_id'].values

start_date = date(1, 1, 1)
end_date = date(25, 12, 31)
n_days_train = (end_date - start_date).days + 1


#Get standard scaler from training data
features = []
for basin_id in basin_list:
    file_path = f'data/baseline/regional_lstm_simp_hymod/processed_lstm_train_datasets/{pb}/lstm_input{basin_id}.csv'
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data = data.drop(columns=['date'])
        features.append(data.iloc[:n_days_train, :29].values) # 29 features
features = np.vstack(features).astype(np.float32) #stack all basins

#standardize features
scaler = StandardScaler()
scaler.fit(features)

####################################################################################################################################################################################################################################
scenario_list = ['scenario3', 'scenario7', 'scenario11', 'scenario15']
for scenario in scenario_list:
    for basin_id in basin_list:
        # for coverage in coverage:
        for coverage in np.append(np.arange(12), [99]):
            # for comb in comb:
            for comb in np.arange(12):
                file_path = f'data/baseline/regional_lstm_simp_hymod/processed_lstm_prediction_datasets/{pb}/lstm_input{basin_id}_coverage{coverage}_comb{comb}.csv'
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path)
                    #replace columns: 'noisy_precip', 'era5temp', 'qobs' with climate scenario values
                    new_precip = pd.read_csv(f'data/{scenario}/noisy_precip/future_noisy_precip{basin_id}_coverage{coverage}_comb{comb}.csv')['PRECIP']
                    new_temp = pd.read_csv(f'data/{scenario}/temperature/future_temp{basin_id}.csv')['tavg']
                    new_qobs = pd.read_csv(f'output/{scenario}/gr4j_true/gr4j_true{basin_id}.csv')['streamflow']
                    data['noisy_precip'] = new_precip
                    data['era5temp'] = new_temp
                    data['qobs_err0r'] = new_qobs

                    grab_date = data['date'].values
                    #keep date only after sequence length
                    grab_date = grab_date[SEQUENCE_LENGTH-1:]

                    data = data.drop(columns=['date'])
                    features = data.iloc[:, :29].values
                    targets = data.iloc[:, [-1]].values

                    #standardize features with training data scaler
                    features = scaler.transform(features)

                    #create sequences of features and targets
                    x_seq, y_seq = create_sequences(features, targets, SEQUENCE_LENGTH)

                    test_dataset = SeqDataset(x_seq, y_seq)
                    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

                    model = LSTMModel(NUM_INPUT_FEATURES, NUM_HIDDEN_NEURONS, NUM_HIDDEN_LAYERS, NUM_OUTPUT_FEATURES, DROPOUT_RATE).to(device)
                    model.load_state_dict(torch.load(f'regional_lstm_simp_hymod{pb}.pth', weights_only=True))
                    model.eval()

                    all_outputs, all_targets = [], []
                    with torch.no_grad():
                        for inputs, target in test_loader:
                            inputs, target = inputs.to(device).float(), target.to(device).float()
                            output = model(inputs)
                            all_outputs.append(output.cpu().numpy())
                            all_targets.append(target.cpu().numpy())
                    all_outputs = np.concatenate(all_outputs).flatten()
                    all_targets = np.concatenate(all_targets).flatten()

                    #save outputs with corresponding dates for the basin
                    temp_df = pd.DataFrame({'date': grab_date, 'true_streamflow': all_targets, 'sim_error': all_outputs})
                    

                    # get hymod flow and rename for clarity
                    simp_hymod_simflow = pd.read_csv(
                        f'output/{scenario}/simp_hymod/simp_hymod{basin_id}_coverage{coverage}_comb{comb}.csv'
                    )[['date', 'streamflow']].rename(columns={'streamflow': 'streamflow_hymod'})

                    # merge by date
                    temp_df = temp_df.merge(simp_hymod_simflow, on='date')

                    # add hymod streamflow with sim_error
                    temp_df['sim_streamflow'] = temp_df['sim_error'] + temp_df['streamflow_hymod']

                    # drop unnecessary columns
                    temp_df = temp_df[['date', 'true_streamflow', 'sim_streamflow']]

                    #round streamflow and true_streamflow to 3 decimal places
                    temp_df = temp_df.round(3)
                    # Save the dataframe to CSV
                    output_file_path = f'output/{scenario}/regional_lstm_simp_hymod/lstm_input{basin_id}_coverage{coverage}_comb{comb}.csv'
                    temp_df.to_csv(output_file_path, index=False)
                    print(f'Scenario: {scenario} | Output saved to {output_file_path}')

                    # clear memory
                    del model
                    del test_dataset, test_loader, all_outputs, all_targets, x_seq, y_seq, features, targets, data, temp_df
                    gc.collect()
                    torch.cuda.empty_cache()
    
