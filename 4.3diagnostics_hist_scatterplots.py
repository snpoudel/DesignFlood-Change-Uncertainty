#load library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
#read dignostic csv file
df = pd.read_csv('output/diagnostics_validperiod.csv')

#make a scatter plot of grid vs RMSE(PRECIP)
plt.figure(figsize=(6,4))
plt.scatter(df['grid'], df['RMSE(PRECIP)'], color='blue', alpha=0.6)
plt.xlabel('Percentage of gauging stations used')
plt.ylabel('Precipitation RMSE (mm/day)')
#change x ticks labels
plt.xticks([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ['5', '10', '20', '30', '40', '50', '60', '70', '80', '90'])
plt.grid(True, linestyle ='--', alpha = 0.5)
plt.savefig('output/PercentageStation_PrecipRMSE.png', dpi=300)
# Clear the plot
plt.clf()  # Clear the current figure
plt.close() # Close the current figure window



# 1. NSE, KGE, RMSE
nse_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
             value_vars=['NSE(HBV)', 'NSE(RECAL_HBV)','NSE(HYMOD)', 'NSE(LSTM)'])
kge_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
             value_vars=['KGE(HBV)', 'KGE(RECAL_HBV)', 'KGE(HYMOD)', 'KGE(LSTM)'])
rmse_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
             value_vars=['RMSE(HBV)', 'RMSE(RECAL_HBV)', 'RMSE(HYMOD)', 'RMSE(LSTM)'])

# Plot scatter points
fig,axs = plt.subplots(3,1,figsize=(5, 6),sharex=True)
sns.scatterplot(data=nse_melt, x='RMSE(PRECIP)', y='value', hue='variable', alpha=0.6, ax=axs[0])
# Compute and plot LOWESS lines
for variable in nse_melt['variable'].unique():
    subset = nse_melt[nse_melt['variable'] == variable]
    lowess = sm.nonparametric.lowess(subset['value'], subset['RMSE(PRECIP)'], frac=0.8)  # Adjust frac as needed
    axs[0].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
axs[0].set_xlabel('Precipitation RMSE (mm/day)')
axs[0].set_ylabel('Streamflow NSE')
axs[0].grid(True, linestyle ='--', alpha = 0.5)
axs[0].legend(title='', loc='best')
new_labels = ['HBV', 'RECALIBRATED HBV', 'HYMOD', 'LSTM']
for t, l in zip(axs[0].get_legend().texts, new_labels):t.set_text(l)
sns.scatterplot(data=kge_melt, x='RMSE(PRECIP)', y='value', hue='variable', alpha=0.6,ax=axs[1])
# Compute and plot LOWESS lines
for variable in kge_melt['variable'].unique():
    subset = kge_melt[kge_melt['variable'] == variable]
    lowess = sm.nonparametric.lowess(subset['value'], subset['RMSE(PRECIP)'], frac=0.8)  # Adjust frac as needed
    axs[1].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
# Customize the plot
axs[1].set_xlabel('Precipitation RMSE (mm/day)')
axs[1].set_ylabel('Streamflow KGE')
axs[1].grid(True, linestyle ='--', alpha = 0.5)
axs[1].get_legend().remove()

sns.scatterplot(data=rmse_melt, x='RMSE(PRECIP)', y='value', hue='variable', alpha=0.6,ax=axs[2])
# Compute and plot LOWESS lines
for variable in rmse_melt['variable'].unique():
    subset = rmse_melt[rmse_melt['variable'] == variable]
    lowess = sm.nonparametric.lowess(subset['value'], subset['RMSE(PRECIP)'], frac=0.8)  # Adjust frac as needed
    axs[2].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
# Customize the plot
axs[2].set_xlabel('Precipitation RMSE (mm/day)')
axs[2].set_ylabel('Streamflow RMSE')
axs[2].grid(True, linestyle ='--', alpha = 0.5)
axs[2].get_legend().remove()
#save the plot
plt.tight_layout()
plt.savefig('output/scatterplot_nse_kge_rmse.png', dpi=300)
# Clear the plot
plt.clf()  # Clear the current figure



# 2. BIAS, HFB
bias_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
             value_vars=['BIAS(HBV)', 'BIAS(RECAL_HBV)', 'BIAS(HYMOD)', 'BIAS(LSTM)'])
hfb_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
             value_vars=['HFB(HBV)', 'HFB(RECAL_HBV)', 'HFB(HYMOD)', 'HFB(LSTM)'])

fig,axs = plt.subplots(2,1, figsize=(6, 6), sharex=True)
#Plot scatter points
sns.scatterplot(data=bias_melt, x='RMSE(PRECIP)', y='value', hue='variable', alpha=0.5, ax=axs[0])
# Compute and plot LOWESS lines
for variable in bias_melt['variable'].unique():
    subset = bias_melt[bias_melt['variable'] == variable]
    lowess = sm.nonparametric.lowess(subset['value'], subset['RMSE(PRECIP)'], frac=0.5)  # Adjust frac as needed
    axs[0].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
# Customize the plot
axs[0].set_xlabel('Precipitation RMSE (mm/day)')
axs[0].set_ylabel('Streamflow Bias (%)')
axs[0].grid(True, linestyle ='--', alpha = 0.5)
axs[0].get_legend().remove()

sns.scatterplot(data=hfb_melt, x='RMSE(PRECIP)', y='value', hue='variable', alpha=0.6, ax=axs[1])
# Compute and plot LOWESS lines
for variable in hfb_melt['variable'].unique():
    subset = hfb_melt[hfb_melt['variable'] == variable]
    lowess = sm.nonparametric.lowess(subset['value'], subset['RMSE(PRECIP)'], frac=0.5)  # Adjust frac as needed
    axs[1].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
# Customize the plot
axs[1].set_xlabel('Precipitation RMSE (mm/day)')
axs[1].set_ylabel('Streamflow HFB (%)')
axs[1].grid(True, linestyle ='--', alpha = 0.5)
axs[1].legend(title='', loc='best')
new_labels = ['HBV', 'RECALIBRATED HBV', 'HYMOD', 'LSTM']
for t, l in zip(axs[1].get_legend().texts, new_labels):t.set_text(l)
#save the plot
plt.savefig('output/scatterplot_bias_hfb.png', dpi=300)
# Clear the plot
plt.clf()  # Clear the current figure
plt.close() # Close the current figure window