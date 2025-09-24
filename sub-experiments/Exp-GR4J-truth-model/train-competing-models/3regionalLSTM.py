import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import time
import os
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc 
from mpi4py import MPI

# # # #Set up communicator to parallelize job in cluster using MPI
comm = MPI.COMM_WORLD #Get the default communicator object
rank = comm.Get_rank() #Get the rank of the current process
size = comm.Get_size() #Get the total number of processes
# rank = 1
# os.chdir('Z:/MA-Precip-Uncertainty-Exp-Gr4j-Truth')
os.chdir('/home/fs01/sp2596/MA-Precip-Uncertainty-Exp-Gr4j-Truth')
#precip buckets
precip_buckets = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
pb = 'pb' + precip_buckets[rank]

start_time = time.time()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
NUM_INPUT_FEATURES = 29
NUM_OUTPUT_FEATURES = 1
NUM_EPOCHS = 20 #20
NUM_HIDDEN_LAYERS = 1
SEQUENCE_LENGTH = 365 #365
NUM_HIDDEN_NEURONS = 256 #256
BATCH_SIZE = 64
LEARNING_RATE = 0.0001 #0.0001
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
        out = self.fc(out)
        return self.relu(out)

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
    file_path = f'data/baseline/regional_lstm/processed_lstm_train_datasets/{pb}/lstm_input{basin_id}.csv'
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data = data.drop(columns=['date'])
        features.append(data.iloc[:n_days_train, :29].values) # 29 features
features = np.vstack(features).astype(np.float32) #stack all basins

#standardize features
scaler = StandardScaler()
scaler.fit(features)

# Load and preprocess data
features, targets = [], []
for basin_id in basin_list:
    file_path = f'data/baseline/regional_lstm/processed_lstm_train_datasets/{pb}/lstm_input{basin_id}.csv'
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data = data.drop(columns=['date'])
        # change qobs to qobs from gr4j true
        data['qobs'] = pd.read_csv(f'output/baseline/gr4j_true/gr4j_true{basin_id}.csv')['streamflow']

        features.append(data.iloc[:n_days_train, :29].values) # 29 features
        targets.append(data.iloc[:n_days_train, [-1]].values) # target is the last column
features = np.vstack(features).astype(np.float32)
targets = np.vstack(targets).astype(np.float32)

#standardize features with training data scaler
features = scaler.transform(features)

#create sequences of features and targets
# x_seq, y_seq = create_sequences(features, targets, SEQUENCE_LENGTH)
#create sequences for each basin and concatenate, this is done so sequence from different basins are not mixed
#first, calculate total number of basins present in this precip bucket
total_num_basins = os.listdir(f'data/baseline/regional_lstm/processed_lstm_train_datasets/{pb}/')
x_seq, y_seq = [], []
for j in range(len(total_num_basins)):
    x_seq_temp, y_seq_temp = create_sequences(features[j*n_days_train:(j+1)*n_days_train], targets[j*n_days_train:(j+1)*n_days_train], SEQUENCE_LENGTH)
    x_seq.append(x_seq_temp)
    y_seq.append(y_seq_temp)
x_seq = np.vstack(x_seq) 
y_seq = np.vstack(y_seq)


# Dataset and DataLoader
train_dataset = SeqDataset(x_seq, y_seq)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = LSTMModel(NUM_INPUT_FEATURES, NUM_HIDDEN_NEURONS, NUM_HIDDEN_LAYERS, NUM_OUTPUT_FEATURES, DROPOUT_RATE).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()
        if (i+1) % 1000 == 0:
            print(f'PB:{pb}, Epoch: [{epoch+1}/{NUM_EPOCHS}], Step: [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f} seconds')

# Save the model
torch.save(model.state_dict(), f'regional_lstm{pb}.pth')
#time taken to train the model
end_time = time.time()
print(f'Time taken for training {pb}: {end_time - start_time:.2f} seconds')
#memory cleanup
gc.collect() #garbage collection
torch.cuda.empty_cache() #free unused memory

# basin_list = ['01096000']
####################################################################################################################################################################################################################################
#######---PREDICTION FOR HISTORICAL PERIOD---#######
for basin_id in basin_list:
    # for coverage in coverage:
    for coverage in np.append(np.arange(12), [99]):
        # for comb in comb:
        for comb in np.arange(12):
            file_path = f'data/baseline/regional_lstm/processed_lstm_prediction_datasets/{pb}/lstm_input{basin_id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                # change qobs to qobs from gr4j true
                data['qobs'] = pd.read_csv(f'output/baseline/gr4j_true/gr4j_true{basin_id}.csv')['streamflow']
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
                model.load_state_dict(torch.load(f'regional_lstm{pb}.pth', weights_only=True))
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
                temp_df = pd.DataFrame({'date': grab_date, 'true_streamflow': all_targets, 'streamflow': all_outputs})
                #round streamflow and true_streamflow to 3 decimal places
                temp_df = temp_df.round(3)
                # Save the dataframe to CSV
                output_file_path = f'output/baseline/regional_lstm/lstm_input{basin_id}_coverage{coverage}_comb{comb}.csv'
                temp_df.to_csv(output_file_path, index=False)

end_time = time.time()
print(f'Time taken for historical dataset prediction on basin: {end_time - start_time:.2f} seconds')