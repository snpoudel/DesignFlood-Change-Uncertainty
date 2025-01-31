import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def nse(obs, sim):
    return 1 - (np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2))

hbv_true = pd.read_csv('output/hbv_true/hbv_true01108000.csv')
hbv_true = hbv_true[364:].reset_index(drop=True)
h_lstm = pd.read_csv('output/regional_lstm_hymod/final_output/historical/hymod_lstm01108000_coverage1_comb1.csv')


hbv_true[0:1000]
h_lstm[0:1000]
nse(hbv_true['streamflow'][0:20000], h_lstm['hymod_lstm_streamflow'][0:20000])

plt.plot(hbv_true['streamflow'][10000:11050], label='True')
plt.plot(h_lstm['hymod_lstm_streamflow'][10000:11050], label='LSTM')
plt.legend()