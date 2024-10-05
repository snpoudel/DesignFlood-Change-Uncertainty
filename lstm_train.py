#import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np



def train_lstm_model(lstm_train_dataset, chunk_size, precip_bucket):
    '''
    lstm_train_dataset: dataset to train the model
    chunk_size: size/nrows of training dataset for single basin
    '''
    #Hyperparameters
    input_size = lstm_train_dataset.shape[1] -1 #53number of features in input at a time step
    hidden_size = 212 #256 #number of features in the hidden state
    num_layers = 1 #number of lstm blocks (having more layers isn't doing much improvements)
    output_size = 1 #size of prediction/output
    sequence_length = 365 #365length of each sequence
    num_epochs = 30 # 30#number of iterations of complete dataset
    batch_size = 64 #128 #number of samples in one batch
    learning_rate = 0.001 #0.005learning rate
    dropout_prob = 0.4 #dropout probability

    #define the lstm model
    class LSTMModel(nn.Module): #class for lstm model
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob): #initialize the lstm model
            super(LSTMModel, self).__init__() #initialize the lstm model module 
            self.hidden_size = hidden_size #hidden size of lstm layer
            self.num_layers = num_layers #number of lstm layers
            #LSTM layer with dropout
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                bidirectional=False) #lstm layer with batch first input format, dropout and bidirectional set to false 
            self.fc = nn.Linear(hidden_size, output_size) #fully connected layer to map lstm output to output size
            self.dropout = nn.Dropout(dropout_prob)
            self.relu = nn.ReLU()

        def forward(self, x): #forward pass through lstm model
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initialize hidden state 
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initialize cell state 

            out,_ = self.lstm(x, (h0, c0)) #this outputs a tuple (output, (h_n,c_n)) where the first element is the output of the lstm layer and the second element is a tuple containing the hidden and cell state of the lstm layer 
            #out = self.fc(out[:, -1, :]) #this maps the output of the lstm layer to the output size and selects the output of the last time step   
            out = out[:, -1, :] # out (batch_size, num_hidden_neuron) for the last hidden
            out = self.dropout(out)
            out = self.fc(out)
            out = self.relu(out)
            return out


    #Load and preprocess data
    data = lstm_train_dataset #loop through and merge the datasets here!!!
    features = data.iloc[:, :-1].values # 2d array of features, everything is features exccept last column
    labels = data.iloc[:, -1].values # 1d array of labels, last column is label

    #normalize features
    scaler = StandardScaler() #standardize features by removing the mean and scaling to unit variance
    features = scaler.fit_transform(features) #fit to data, then transform it

    #create sequences and targets from features and labels 
    def create_sequences(features, labels, sequence_length): #function to create sequences
        sequences = []
        targets = []
        for i in range(len(features)-sequence_length): #loop through the features 
            seq = features[i:i+sequence_length] #get the sequence
            label = labels[i+sequence_length] #get the label
            sequences.append(seq) #append the sequence
            targets.append(label) #append the label
        return np.array(sequences), np.array(targets) #return the sequences and targets as numpy arrays

    #make sequences for each basin and than merge'em
    for i in np.arange(0, len(features), chunk_size):
        features_chunk = features[i:i+chunk_size] #get the chunk of features
        labels_chunk = labels[i:i+chunk_size] #get the chunk of labels
        if i == 0: #if it is the first chunk
            sequences, targets = create_sequences(features_chunk, labels_chunk, sequence_length) #create sequences and targets from the chunk
        else: #if it is not the first chunk
            sequences_chunk, targets_chunk = create_sequences(features_chunk, labels_chunk, sequence_length) #create sequences and targets from the chunk
            sequences = np.concatenate((sequences, sequences_chunk), axis=0) #concatenate the sequences
            targets = np.concatenate((targets, targets_chunk), axis=0) #concatenate the targets

    #convert data to pytorch tensors 
    sequences = torch.tensor(sequences, dtype=torch.float32) #convert sequences to tensor
    targets = torch.tensor(targets, dtype=torch.float32).view(-1,1) #convert targets to tensor and reshape it to 2d tensor 

    #create train and test sets and data loaders
    train_dataset = torch.utils.data.TensorDataset(sequences, targets) #create a dataset from sequences and targets 
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #create a data loader for training data

    #initialize lstm model, loss function and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to(device) #initialize lstm model 
    criterion = nn.MSELoss() #initialize loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #initialize optimizer 

    #train the model
    for epoch in range(num_epochs): #loop through the dataset for number of epochs
        model.train() #set the model to training mode, this is not necessary for this model because we are not using dropout or batch normalization, but it is good practice to set the model to training mode
        for i, (inputs, targets) in enumerate(train_loader): #loop through the training data
            inputs, targets =inputs.to(device), targets.to(device)

            #forward pass
            outputs = model(inputs) #forward pass through lstm model
            loss = criterion(outputs, targets) #calculate loss

            #backward pass
            optimizer.zero_grad() #zero gradient
            loss.backward() #calculate gradient
            optimizer.step() #update weights

            if (i+1) % 100 == 0: #print loss every 20 steps
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    #save the trained model
    torch.save(model.state_dict(), f'trained_lstm_model_{precip_bucket}.pth')
    print('Model Training Completed!!!')
