import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Function to make non exceedance probability plot
def make_nep_plot(data_hist, data_future):
    sorted_data = np.sort(data_hist)
    sorted_probability = np.arange(1,len(sorted_data)+1)/len(sorted_data)
    plt.figure(figsize=(6,4))
    plt.plot(sorted_probability, sorted_data, marker='o', linestyle='-.',
             color='blue', markersize=0, linewidth=2, label='Historical', alpha=0.8)
    sorted_data = np.sort(data_future)
    sorted_probability = np.arange(1,len(sorted_data)+1)/len(sorted_data)
    #make y axis log scale that works of 0 values
    #plt.yscale('symlog', linthresh=0.1)
    plt.yscale('log')
    plt.plot(sorted_probability, sorted_data, marker='o', linestyle='-',
             color='red', markersize=0, linewidth=2, label='Future', alpha=0.8)
    plt.ylim(0.01,None)
    plt.xlim(0.2,None)
    plt.xlabel('Non-Exceedance Probability')
    plt.ylabel('MAP (mm/day)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

#Load data
df_hist = pd.read_csv("data/true_precip/true_precip01108000.csv")
df_future = pd.read_csv('data/future/future_true_precip/future_true_precip01108000.csv')

#Make the plot
make_nep_plot(data_hist=df_hist['PRECIP'], data_future=df_future['PRECIP'])

#Save the plot
plt.savefig('output/nep_plot_logscale.png', dpi=300)