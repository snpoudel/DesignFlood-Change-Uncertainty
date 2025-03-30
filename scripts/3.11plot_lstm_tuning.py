import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')

#For streamflow predicting LSTM model
# Load the results from the hyperparameter tuning
df = pd.DataFrame()
for i in range(32):
    file = pd.read_csv(f'output/lstm_tuning/streamflow_model/lstm_tuning_{i}.csv')
    df = pd.concat([df, file], axis = 0).reset_index(drop=True)
#find mean train and validation loss for k folds
df_mean = df.groupby(['num_epochs', 'num_hidden_neurons', 'batch_size', 'learning_rate', 'dropout_rate']).mean().reset_index()
df_mean= df_mean.drop(columns=['kfold'])
df_mean = df_mean.sort_values('val_loss', ascending = True).reset_index(drop=True)
df_mean


# Plot num epoch vs val loss, num_hidden_neurons vs val loss, batch size vs val loss, learning rate vs val loss, dropout rate vs val loss
fig, ax = plt.subplots(2, 3, figsize=(6, 5))
# Plotting with more informative and nice way
sns.boxplot(x='num_epochs', y='val_loss', data=df_mean, ax=ax[0, 0], width=0.5)
# ax[0, 0].set_title('Num Epochs vs Validation Loss')
ax[0, 0].set_xlabel('Num Epochs')
ax[0, 0].set_ylabel('4-Fold Cross-Validation MSE')
sns.boxplot(x='num_hidden_neurons', y='val_loss', data=df_mean, ax=ax[0, 1], width=0.5)
# ax[0, 1].set_title('Num Hidden Neurons vs Validation Loss')
ax[0, 1].set_xlabel('Num Hidden Neurons')
ax[0, 1].set_ylabel('4-Fold Cross-Validation MSE')
sns.boxplot(x='batch_size', y='val_loss', data=df_mean, ax=ax[0, 2], width=0.5)
# ax[0, 2].set_title('Batch Size vs Validation Loss')
ax[0, 2].set_xlabel('Batch Size')
ax[0, 2].set_ylabel('4-Fold Cross-Validation MSE')
sns.boxplot(x='learning_rate', y='val_loss', data=df_mean, ax=ax[1, 0], width=0.5)
# ax[1, 0].set_title('Learning Rate vs Validation Loss')
ax[1, 0].set_xlabel('Learning Rate')
ax[1, 0].set_ylabel('4-Fold Cross-Validation MSE')
sns.boxplot(x='dropout_rate', y='val_loss', data=df_mean, ax=ax[1, 1], width=0.5)
# ax[1, 1].set_title('Dropout Rate vs Validation Loss')
ax[1, 1].set_xlabel('Dropout Rate')
ax[1, 1].set_ylabel('4-Fold Cross-Validation MSE')
# Remove the empty subplot
fig.delaxes(ax[1, 2])
# Add text to the plot
best_params = df_mean.iloc[0]
best_text = f"Best Params:\nEpochs: {best_params['num_epochs']}\nNeurons: {best_params['num_hidden_neurons']}\nBatch Size: {best_params['batch_size']}\nLearning Rate: {best_params['learning_rate']}\nDropout Rate: {best_params['dropout_rate']}\nVal RMSE: {best_params['val_loss']:.3f}"
fig.text(0.7, 0.3, best_text, fontsize=11, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
fig.suptitle('Streamflow Prediction LSTM Model Hyperparameter Tuning')
plt.tight_layout()
plt.savefig('output/figures/streamflow_lstm_tuning_results.png', dpi=300)
plt.show()


#For residual predicting LSTM model
# Load the results from the hyperparameter tuning
df = pd.DataFrame()
for i in range(32):
    file = pd.read_csv(f'output/lstm_tuning/residual_model/lstm_tuning_{i}.csv')
    df = pd.concat([df, file], axis = 0).reset_index(drop=True)
#find mean train and validation loss for k folds
df_mean = df.groupby(['num_epochs', 'num_hidden_neurons', 'batch_size', 'learning_rate', 'dropout_rate']).mean().reset_index()
df_mean= df_mean.drop(columns=['kfold'])
df_mean = df_mean.sort_values('val_loss', ascending = True).reset_index(drop=True)
df_mean

# Plot num epoch vs val loss, num_hidden_neurons vs val loss, batch size vs val loss, learning rate vs val loss, dropout rate vs val loss
fig, ax = plt.subplots(2, 3, figsize=(6, 5))
# Plotting with more informative and nice way
sns.boxplot(x='num_epochs', y='val_loss', data=df_mean, ax=ax[0, 0], width=0.5)
# ax[0, 0].set_title('Num Epochs vs Validation Loss')
ax[0, 0].set_xlabel('Num Epochs')
ax[0, 0].set_ylabel('4-Fold Cross-Validation MSE')
sns.boxplot(x='num_hidden_neurons', y='val_loss', data=df_mean, ax=ax[0, 1], width=0.5)
# ax[0, 1].set_title('Num Hidden Neurons vs Validation Loss')
ax[0, 1].set_xlabel('Num Hidden Neurons')
ax[0, 1].set_ylabel('4-Fold Cross-Validation MSE')
sns.boxplot(x='batch_size', y='val_loss', data=df_mean, ax=ax[0, 2], width=0.5)
# ax[0, 2].set_title('Batch Size vs Validation Loss')
ax[0, 2].set_xlabel('Batch Size')
ax[0, 2].set_ylabel('4-Fold Cross-Validation MSE')
sns.boxplot(x='learning_rate', y='val_loss', data=df_mean, ax=ax[1, 0], width=0.5)
# ax[1, 0].set_title('Learning Rate vs Validation Loss')
ax[1, 0].set_xlabel('Learning Rate')
ax[1, 0].set_ylabel('4-Fold Cross-Validation MSE')
sns.boxplot(x='dropout_rate', y='val_loss', data=df_mean, ax=ax[1, 1], width=0.5)
# ax[1, 1].set_title('Dropout Rate vs Validation Loss')
ax[1, 1].set_xlabel('Dropout Rate')
ax[1, 1].set_ylabel('4-Fold Cross-Validation MSE')
# Remove the empty subplot
fig.delaxes(ax[1, 2])
# Add text to the plot
best_params = df_mean.iloc[0]
best_text = f"Best Params:\nEpochs: {best_params['num_epochs']}\nNeurons: {best_params['num_hidden_neurons']}\nBatch Size: {best_params['batch_size']}\nLearning Rate: {best_params['learning_rate']}\nDropout Rate: {best_params['dropout_rate']}\nVal RMSE: {best_params['val_loss']:.3f}"
fig.text(0.7, 0.3, best_text, fontsize=11, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
fig.suptitle('Process Model Residual Prediction LSTM Model Hyperparameter Tuning')
plt.tight_layout()
plt.savefig('output/figures/residual_lstm_tuning_results.png', dpi=300)
plt.show()