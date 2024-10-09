#import libraries
import torch
import torch.nn as nn
import numpy as np


def predict_lstm_model(lstm_predict_dataset, precip_bucket):
    '''
    lstm_predict_dataset: dataset to predict with trained lstm model
    '''
    #Hyperparameters
    input_size = lstm_predict_dataset.shape[1] #53number of features in input at a time step
    hidden_size = 256 #256 #number of features in the hidden state
    num_layers = 1 #number of lstm blocks (having more layers isn't doing much improvements)
    output_size = 1 #size of prediction/output
    sequence_length = 365 #365length of each sequence
    num_epochs = 60 # 30#number of iterations of complete dataset
    batch_size = 128 #128 #number of samples in one batch
    learning_rate = 0.0005 #0.005learning rate
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

    # Load the trained model
    def load_model(input_size, hidden_size, num_layers, output_size, dropout_prob):
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)
        model.load_state_dict(torch.load(f'trained_lstm_model_{precip_bucket}.pth'))
        model.eval()
        return model

    #create sequences and targets from features and labels 
    def create_sequences(features, sequence_length): #function to create sequences
        sequences = []
        for i in range(len(features)-sequence_length): #loop through the features 
            seq = features[i:i+sequence_length] #get the sequence
            sequences.append(seq) #append the sequence
        return np.array(sequences) #return the sequences as numpy arrays
    sequences = create_sequences(lstm_predict_dataset, sequence_length)
    
    # #convert data to pytorch tensors 
    sequences = torch.tensor(sequences, dtype=torch.float32) #convert sequences to tensor
   
    #Load the trained lstm model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(input_size, hidden_size, num_layers, output_size, dropout_prob).to(device)
  
    model_output_comb = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            inputs = sequences[i:i+batch_size, :, :]
            inputs = inputs.to(device)
            model_output = model(inputs).cpu().numpy()
            model_output = np.round(model_output.flatten(), 3)
            model_output_comb= np.concatenate([model_output_comb, model_output])
    return model_output_comb #returns lstm model predictions
    