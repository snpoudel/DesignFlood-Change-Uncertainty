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

start_time = time.time()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#use kfold cross validation to tune hyperparameters
kfold = 4
#make a dictionary of hyperparameters to tune
#hyperparameters to tune: num_epochs, num_hidden_neurons, batch_size, learning_rate, dropout_rate
num_input_features = 29
num_output_features = 1
num_epochs = [20, 30]
num_hidden_layers = 1
num_hidden_neurons = [128, 256]
sequence_length = 365
batch_size = [32, 64]
learning_rate = [0.0001, 0.0005]
dropout_rate = [0.2, 0.4]

# num_input_features = 29
# num_output_features = 1
# num_epochs = [2, 3]
# num_hidden_layers = 1
# num_hidden_neurons = [4, 5]
# sequence_length = 3
# batch_size = [32, 64]
# learning_rate = [0.01, 0.02]
# dropout_rate = [0.2, 0.4]

#make dataframe with all possible combinations of hyperparameters
# Initialize a list to store rows
rows = []
# Loop to generate hyperparameter combinations
for num_epoch in num_epochs:
    for num_hidden_neuron in num_hidden_neurons:
        for batch in batch_size:
            for lr in learning_rate:
                for dropout in dropout_rate:
                    rows.append({
                        'num_input_features': num_input_features,
                        'num_output_features': num_output_features,
                        'num_epochs': num_epoch,
                        'num_hidden_layers': num_hidden_layers,
                        'num_hidden_neurons': num_hidden_neuron,
                        'sequence_length': sequence_length,
                        'batch_size': batch,
                        'learning_rate': lr,
                        'dropout_rate': dropout
                    })
# Create the DataFrame in one step
df_hp = pd.DataFrame(rows)

#loop through all combinations of hyperparameters
#initialize dataframe to store results
df = pd.DataFrame(columns=['kfold', 'num_epochs', 'num_hidden_neurons', 'batch_size', 'learning_rate', 'dropout_rate', 'train_loss', 'val_loss'])
# for iter in range(len(df_hp)):  
iter = rank                 
# Hyperparameters
NUM_INPUT_FEATURES = int(df_hp['num_input_features'][iter])
NUM_OUTPUT_FEATURES = int(df_hp['num_output_features'][iter])
NUM_EPOCHS = int(df_hp['num_epochs'][iter])
NUM_HIDDEN_LAYERS = int(df_hp['num_hidden_layers'][iter])
NUM_HIDDEN_NEURONS = int(df_hp['num_hidden_neurons'][iter])
SEQUENCE_LENGTH = int(df_hp['sequence_length'][iter])
BATCH_SIZE = int(df_hp['batch_size'][iter])
LEARNING_RATE = float(df_hp['learning_rate'][iter])
DROPOUT_RATE = float(df_hp['dropout_rate'][iter])

# Function to create LSTM input sequences
def create_sequences(features, targets, sequence_length):
    n_days = len(features) - sequence_length + 1
    x_sequences, y_sequences = [], []
    for j in range(n_days):
        x_seq = features[j:j+sequence_length]
        y_seq = targets[j+sequence_length-1]
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

# Dates and basins for training
# basin_list = ['01095375', '01096000', '01097000', '01103280', '01104500', '01105000']
basin_list = pd.read_csv("data/regional_lstm/MA_basins_gauges_2000-2020_filtered.csv", dtype={'basin_id':str})['basin_id'].values

start_date = date(2000, 1, 1)
end_date = date(2013, 12, 31)
n_days_train = (end_date - start_date).days + 1


#Get standard scaler from training data
features = []
for basin_id in basin_list:
    file_path = f'data/regional_lstm/processed_lstm_input/pb0/lstm_input{basin_id}.csv'
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
    file_path = f'data/regional_lstm/processed_lstm_input/pb0/lstm_input{basin_id}.csv'
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
#split x_seq, y_seq into 4 folds for cross validation
x_seq = np.array_split(x_seq, kfold)
y_seq = np.array_split(y_seq, kfold)

#train model for kfold-1 folds and validate on the remaining fold
rows_temp = [] #store results for each fold
for i in range(kfold): #use ith fold as validation set
    x_val = x_seq[i] 
    y_val = y_seq[i]
    x_train = np.concatenate([fold for k, fold in enumerate(x_seq) if k != i], axis=0)
    y_train = np.concatenate([fold for k, fold in enumerate(y_seq) if k != i], axis=0)

    # Dataset and DataLoader
    train_dataset = SeqDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    valid_dataset = SeqDataset(x_val, y_val)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = LSTMModel(NUM_INPUT_FEATURES, NUM_HIDDEN_NEURONS, NUM_HIDDEN_LAYERS, NUM_OUTPUT_FEATURES, DROPOUT_RATE).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        avg_train_loss = 0
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.time()
            if iter % 5 == 0:
                if (idx+1) % 1000 == 0:
                    print(f'Rank: {iter}, Kfold:[{i}/{kfold}] Epoch: [{epoch+1}/{NUM_EPOCHS}], Step: [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f} seconds')
            avg_train_loss += loss.item()
        avg_train_loss /= len(train_loader)

    # Validation Loss
    #find validation loss on entire validation set, ith fold
    with torch.no_grad():
        avg_val_loss = 0
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            val_loss = loss_fn(predictions, targets)
            #convert to numpy value
            val_loss = val_loss.cpu().numpy()
            avg_val_loss += val_loss
        avg_val_loss /= len(valid_loader)
    
    #save results to dataframe
    rows_temp.append({
        'kfold': i,
        'num_epochs': NUM_EPOCHS,
        'num_hidden_neurons': NUM_HIDDEN_NEURONS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'dropout_rate': DROPOUT_RATE,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss
    })
    df = pd.DataFrame(rows_temp)
    #save dataframe to csv
    df.to_csv(f'output/lstm_tuning/streamflow_model/lstm_tuning_{iter}.csv', index=False)
print(f'Finished training for hyperparameter combination {iter} in {time.time() - start_time:.2f} seconds')