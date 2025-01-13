import numpy as np
import pandas as pd
from datetime import date
import time
import os
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mpi4py import MPI

#Set up communicator to parallelize job in cluster using MPI
comm = MPI.COMM_WORLD #Get the default communicator object
rank = comm.Get_rank() #Get the rank of the current process
size = comm.Get_size() #Get the total number of processes

#precip buckets
precip_buckets = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
pb = 'pb' + precip_buckets[rank]

start_time = time.time()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
NUM_INPUT_FEATURES = 29
NUM_OUTPUT_FEATURES = 1
NUM_EPOCHS = 30
NUM_HIDDEN_LAYERS = 1
SEQUENCE_LENGTH = 365
NUM_HIDDEN_NEURONS = 256
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.4

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
basin_list = pd.read_csv("data/regional_lstm/MA_basins_gauges_2000-2020_filtered.csv", dtype={'basin_id':str})['basin_id'].values

start_date = date(2000, 1, 1)
end_date = date(2013, 12, 31)
n_days_train = (end_date - start_date).days + 1


#Get standard scaler from training data
features = []
for basin_id in basin_list:
    file_path = f'data/regional_lstm/processed_lstm_input/{pb}/lstm_input{basin_id}.csv'
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
    file_path = f'data/regional_lstm/processed_lstm_input/{pb}/lstm_input{basin_id}.csv'
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data = data.drop(columns=['date'])
        features.append(data.iloc[:n_days_train, :29].values) # 29 features
        targets.append(data.iloc[:n_days_train, [-1]].values) # target is the last column
features = np.vstack(features).astype(np.float32)
targets = np.vstack(targets).astype(np.float32)

#standardize features with training data scaler
features = scaler.transform(features)

#create sequences of features and targets
x_seq, y_seq = create_sequences(features, targets, SEQUENCE_LENGTH)

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
        if (i+1) % 500 == 0:
            print(f'PB:{pb}, Epoch: [{epoch+1}/{NUM_EPOCHS}], Step: [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f} seconds')

# Save the model
torch.save(model.state_dict(), f'trained_lstm_model_{pb}.pth')
#time taken to train the model
end_time = time.time()
print(f'Time taken for training: {end_time - start_time:.2f} seconds')

#######---PREDICTION FOR HISTORICAL PERIOD---#######
#### make prediction using trained model for historical period
start_date = date(2000, 1, 1)
end_date = date(2020, 12, 31)
n_days_test = (end_date - start_date).days + 1

features, targets = [], []
for basin_id in basin_list:
    for coverage in np.append(np.arange(12), [99]):
        for comb in np.arange(12):
            file_path = f'data/regional_lstm/prediction_datasets/historical/{pb}/lstm_input{basin_id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                data = data.drop(columns=['date'])
                features.append(data.iloc[:n_days_test, :29].values)
                targets.append(data.iloc[:n_days_test, [-1]].values)
features = np.vstack(features).astype(np.float32)
targets = np.vstack(targets).astype(np.float32)

#standardize features with training data scaler
features = scaler.transform(features)

#create sequences of features and targets
x_seq, y_seq = create_sequences(features, targets, SEQUENCE_LENGTH)

test_dataset = SeqDataset(x_seq, y_seq)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMModel(NUM_INPUT_FEATURES, NUM_HIDDEN_NEURONS, NUM_HIDDEN_LAYERS, NUM_OUTPUT_FEATURES, DROPOUT_RATE).to(device)
model.load_state_dict(torch.load(f'trained_lstm_model_{pb}.pth', weights_only=True))
model.eval()

all_outputs, all_targets = [], []
with torch.no_grad():
    for inputs, target in test_loader:
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        all_outputs.append(output.cpu().numpy())
        all_targets.append(target.cpu().numpy())
all_outputs = np.concatenate(all_outputs).flatten()
all_targets = np.concatenate(all_targets).flatten()

# print(all_outputs.shape, all_targets.shape)

#save outputs with corresponding dates for each basin
i = 0
for basin_id in basin_list:
    for coverage in np.append(np.arange(12), [99]):
        for comb in np.arange(12):
            file_path = f'data/regional_lstm/prediction_datasets/historical/{pb}/lstm_input{basin_id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                if i == 0: #first basin
                    first_n_days_test = n_days_test - SEQUENCE_LENGTH + 1
                    prediction_start_date = date(2000, 1, 1) + pd.DateOffset(days=SEQUENCE_LENGTH-1)
                    test_basin_outputs = all_outputs[i * first_n_days_test:(i + 1) * first_n_days_test]
                    test_basin_targets = all_targets[i * first_n_days_test:(i + 1) * first_n_days_test]
                    date_range = pd.date_range(prediction_start_date, end_date)[:len(test_basin_outputs)]
                    temp_df = pd.DataFrame({'date': date_range, 'true_streamflow': test_basin_targets, 'streamflow': test_basin_outputs})

                else: #other basins
                    test_basin_outputs = all_outputs[i * n_days_test:(i + 1) * n_days_test]
                    test_basin_targets = all_targets[i * n_days_test:(i + 1) * n_days_test]
                    date_range = pd.date_range(prediction_start_date, end_date)
                    temp_df = pd.DataFrame({'streamflow': test_basin_outputs,'true_streamflow': test_basin_targets})
                    #only keep upto date range
                    temp_df = temp_df[:len(date_range)]
                    #add date to the dataframe after sequence length
                    temp_df['date'] = pd.date_range(prediction_start_date, end_date)
                    
                # Save the dataframe to CSV
                output_file_path = f'output/regional_lstm/historical/lstm_input{basin_id}_coverage{coverage}_comb{comb}.csv'
                temp_df.to_csv(output_file_path, index=False)
                i += 1

end_time = time.time()
print(f'Time taken for historical dataset prediction: {end_time - start_time:.2f} seconds')



#######---PREDICTION FOR Future PERIOD---#######
#### make prediction using trained model for future period
start_date = date(2000, 1, 1)
end_date = date(2020, 12, 31)
n_days_test = (end_date - start_date).days + 1

features, targets = [], []
for basin_id in basin_list:
    for coverage in np.append(np.arange(12), [99]):
        for comb in np.arange(12):
            file_path = f'data/regional_lstm/prediction_datasets/future/{pb}/lstm_input{basin_id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                data = data.drop(columns=['date'])
                features.append(data.iloc[:n_days_test, :29].values)
                targets.append(data.iloc[:n_days_test, [-1]].values)
features = np.vstack(features).astype(np.float32)
targets = np.vstack(targets).astype(np.float32)

#standardize features with training data scaler
features = scaler.transform(features)

#create sequences of features and targets
x_seq, y_seq = create_sequences(features, targets, SEQUENCE_LENGTH)

test_dataset = SeqDataset(x_seq, y_seq)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMModel(NUM_INPUT_FEATURES, NUM_HIDDEN_NEURONS, NUM_HIDDEN_LAYERS, NUM_OUTPUT_FEATURES, DROPOUT_RATE).to(device)
model.load_state_dict(torch.load(f'trained_lstm_model_{pb}.pth', weights_only=True))
model.eval()

all_outputs, all_targets = [], []
with torch.no_grad():
    for inputs, target in test_loader:
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        all_outputs.append(output.cpu().numpy())
        all_targets.append(target.cpu().numpy())
all_outputs = np.concatenate(all_outputs).flatten()
all_targets = np.concatenate(all_targets).flatten()

# print(all_outputs.shape, all_targets.shape)

#save outputs with corresponding dates for each basin
i = 0
for basin_id in basin_list:
    for coverage in np.append(np.arange(12), [99]):
        for comb in np.arange(12):
            file_path = f'data/regional_lstm/prediction_datasets/future/{pb}/lstm_input{basin_id}_coverage{coverage}_comb{comb}.csv'
            if os.path.exists(file_path):
                if i == 0: #first basin
                    first_n_days_test = n_days_test - SEQUENCE_LENGTH + 1
                    prediction_start_date = date(2000, 1, 1) + pd.DateOffset(days=SEQUENCE_LENGTH-1)
                    test_basin_outputs = all_outputs[i * first_n_days_test:(i + 1) * first_n_days_test]
                    test_basin_targets = all_targets[i * first_n_days_test:(i + 1) * first_n_days_test]
                    date_range = pd.date_range(prediction_start_date, end_date)[:len(test_basin_outputs)]
                    temp_df = pd.DataFrame({'date': date_range, 'true_streamflow': test_basin_targets, 'streamflow': test_basin_outputs})

                else: #other basins
                    test_basin_outputs = all_outputs[i * n_days_test:(i + 1) * n_days_test]
                    test_basin_targets = all_targets[i * n_days_test:(i + 1) * n_days_test]
                    date_range = pd.date_range(prediction_start_date, end_date)
                    temp_df = pd.DataFrame({'streamflow': test_basin_outputs,'true_streamflow': test_basin_targets})
                    #only keep upto date range
                    temp_df = temp_df[:len(date_range)]
                    #add date to the dataframe after sequence length
                    temp_df['date'] = pd.date_range(prediction_start_date, end_date)
                    
                # Save the dataframe to CSV
                output_file_path = f'output/regional_lstm/future/lstm_input{basin_id}_coverage{coverage}_comb{comb}.csv'
                temp_df.to_csv(output_file_path, index=False)
                i += 1

end_time = time.time()
print(f'Time taken for future dataset prediction: {end_time - start_time:.2f} seconds')