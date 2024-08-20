#load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read in the HBV high flow error distribution csv file
df_hbv_error = pd.read_csv('output/99.9hbvflow_error_distribution_validperiod.csv', usecols = ['grid', 'flow_error'])

#read in the HBV recalibrated high flow error distribution csv file
df_hbv_recalib_error = pd.read_csv('output/99.9hbv_recalibflow_error_distribution_validperiod.csv', usecols = ['grid', 'flow_error'])

#read in the HYMOD flow error distribution csv file
df_hymod_error = pd.read_csv('output/99.9hymodflow_error_distribution_validperiod.csv', usecols = ['grid', 'flow_error'])

#read in the LSTM flow error distribution csv file
df_lstm_error = pd.read_csv('output/99.9lstmflow_error_distribution_validperiod.csv', usecols = ['grid', 'flow_error'])

fig,axes = plt.subplots(4,1, figsize=(10,12), sharex = True)
sns.violinplot(data = df_hbv_error, x = 'grid', y = 'flow_error', split = True,
                linewidth = 1.5, fill=True,  color = 'blue', alpha = 0.8, ax = axes[0])
axes[0].set_ylabel('>99.9th flow error(mm/day)')
axes[0].legend(['HBV'], loc='lower right', frameon=False)


sns.violinplot(data = df_hbv_recalib_error, x = 'grid', y = 'flow_error', split = True,
                linewidth = 1.5, fill=True, color = 'red',alpha = 0.8, ax = axes[1])
axes[1].set_ylabel('>99.9th flow error(mm/day)')
axes[1].legend(['HBV recalibrated'], loc='lower right', frameon=False)

sns.violinplot(data = df_hymod_error, x = 'grid', y = 'flow_error', split = True,
               linewidth = 1.5, fill=True, color = 'brown',alpha = 0.8, ax = axes[2])
axes[2].set_ylabel('>99.9th flow error(mm/day)')
axes[2].legend(['HYMOD'], loc = 'lower right', frameon=False)

sns.violinplot(data = df_lstm_error, x = 'grid', y = 'flow_error', split = True,
               linewidth = 1.5, fill=True, color = 'green',alpha = 0.8, ax = axes[3])
axes[3].set_xticklabels(['5%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
axes[3].set_xlabel('Percentage of preciptation gauging station used')
axes[3].set_ylabel('>99.9th flow error(mm/day)')
axes[3].legend(['LSTM'], loc = 'lower right', frameon=False)

#turn on grids
axes[0].grid(True, linestyle = '--', alpha=0.4)
axes[1].grid(True, linestyle = '--', alpha=0.4)
axes[2].grid(True, linestyle = '--', alpha=0.4)
axes[3].grid(True, linestyle = '--', alpha=0.4)
plt.tight_layout()
plt.show()

#save the plot
fig.savefig('output/99.9high_flow_error_distribution_plot.png', dpi=300)