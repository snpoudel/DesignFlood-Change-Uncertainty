#import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def run_lstm_model(lstm_dataset, new_data):
    #Hyperparameters
    input_size = 3 #number of features in input at a time step
    hidden_size = 64 #64 #number of features in the hidden state
    num_layers = 1 #number of lstm blocks (having more layers isn't doing much improvements)
    output_size = 1 #size of prediction/output
    sequence_length = 365 #length of each sequence
    num_epochs = 50 # 100#number of iterations of complete dataset
    batch_size = 24 #128 #number of samples in one batch
    learning_rate = 0.005 # #learning rate
    dropout_prob = 0.4 #dropout probability

    #define the lstm model
    class LSTMModel(nn.Module): #class for lstm model
        def __init__(self, input_size, hidden_size, num_layers, output_size): #initialize the lstm model
            super(LSTMModel, self).__init__() #initialize the lstm model module 
            self.hidden_size = hidden_size #hidden size of lstm layer
            self.num_layers = num_layers #number of lstm layers
            #LSTM layer with dropout
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                dropout=dropout_prob, bidirectional=False) #lstm layer with batch first input format, dropout and bidirectional set to false 
            self.fc = nn.Linear(hidden_size, output_size) #fully connected layer to map lstm output to output size

        def forward(self, x): #forward pass through lstm model
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #initialize hidden state 
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #initialize cell state 

            out,_ = self.lstm(x, (h0, c0)) #this outputs a tuple (output, (h_n,c_n)) where the first element is the output of the lstm layer and the second element is a tuple containing the hidden and cell state of the lstm layer 
            out = self.fc(out[:, -1, :]) #this maps the output of the lstm layer to the output size and selects the output of the last time step   
            return out


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


    #Load and preprocess data
    data = lstm_dataset
    features = data.iloc[:, :-1].values # 2d array of features, everything is features exccept last column
    labels = data.iloc[:, 3].values # 1d array of labels, last column is label

    #normalize features
    scaler = StandardScaler() #standardize features by removing the mean and scaling to unit variance
    features = scaler.fit_transform(features) #fit to data, then transform it

    #create sequences
    sequences, targets = create_sequences(features, labels, sequence_length) #create sequences and targets from features and labels

    #convert data to pytorch tensors 
    sequences = torch.tensor(sequences, dtype=torch.float32) #convert sequences to tensor
    targets = torch.tensor(targets, dtype=torch.float32).view(-1,1) #convert targets to tensor and reshape it to 2d tensor 

    #create train and test sets and data loaders
    dataset = torch.utils.data.TensorDataset(sequences, targets) #create a dataset from sequences and targets 
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.35) #split the dataset into train and test sets

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #create a data loader for training data
    #test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) #create a data loader for test data

    #initialize lstm model, loss function and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers, output_size) #initialize lstm model 
    criterion = nn.MSELoss() #initialize loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #initialize optimizer 

    #train the model
    for epoch in range(num_epochs): #loop through the dataset for number of epochs
        model.train() #set the model to training mode, this is not necessary for this model because we are not using dropout or batch normalization, but it is good practice to set the model to training mode
        for i, (inputs, targets) in enumerate(train_loader): #loop through the training data

            #forward pass
            outputs = model(inputs) #forward pass through lstm model
            loss = criterion(outputs, targets) #calculate loss

            #backward pass
            optimizer.zero_grad() #zero gradient
            loss.backward() #calculate gradient
            optimizer.step() #update weights

            if (i+1) % 50 == 0: #print loss every 20 steps
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')


    #Make predictions using lstm model
    with torch.no_grad(): #no need to calculate gradients
        model_output = model(sequences) #make predictions using lstm model
        model_output = model_output.numpy() #convert model output to numpy array
          

    #Make predictions for new data using lstm model
    data = new_data
    features = data.iloc[:, :-1].values # 2d array of features, everything is features exccept last column
    #normalize features
    scaler = StandardScaler() #standardize features by removing the mean and scaling to unit variance
    features = scaler.fit_transform(features) #fit to data, then transform it
    #create sequences
    sequences,_ = create_sequences(features, labels, sequence_length) #create sequences and targets from features and labels
    #convert data to pytorch tensors 
    sequences = torch.tensor(sequences, dtype=torch.float32) #convert sequences to tensor
    with torch.no_grad():
        new_model_output = model(sequences)
        new_model_output = new_model_output.numpy()

    return model_output, new_model_output #return output of lstm model function