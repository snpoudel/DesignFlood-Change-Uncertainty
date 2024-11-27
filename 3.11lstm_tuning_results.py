import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#For streamflow predicting LSTM model
# Load the results from the hyperparameter tuning
df = pd.DataFrame()
for i in range(32):
    file = pd.read_csv(f'output/lstm_tuning/lstm_tuning_{i}.csv')
    df = pd.concat([df, file], axis = 0).reset_index(drop=True)
#find mean train and validation loss for k folds
df_mean = df.groupby(['num_epochs', 'num_hidden_neurons', 'batch_size', 'learning_rate', 'dropout_rate']).mean().reset_index()
df_mean= df_mean.drop(columns=['kfold'])
df_mean = df_mean.sort_values('val_loss', ascending = True)
df_mean


#plot num epoch vs val loss, num_hidden_neurons vs val loss, batch size vs val loss, learning rate vs val loss, dropout rate vs val loss
fig, ax = plt.subplots(2, 3, figsize=(6, 4))
sns.boxplot(x='num_epochs', y='val_loss', data=df_mean, ax=ax[0, 0], width=0.25)
sns.boxplot(x='num_hidden_neurons', y='val_loss', data=df_mean, ax=ax[0, 1], width=0.25)
sns.boxplot(x='batch_size', y='val_loss', data=df_mean, ax=ax[0, 2], width=0.25)
sns.boxplot(x='learning_rate', y='val_loss', data=df_mean, ax=ax[1, 0], width=0.25)
sns.boxplot(x='dropout_rate', y='val_loss', data=df_mean, ax=ax[1, 1], width=0.25)
plt.tight_layout()
plt.show()

#For residual predicting LSTM model