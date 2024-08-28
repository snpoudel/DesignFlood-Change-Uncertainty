import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
flood = pd.read_csv('output/tyr_flood.csv')
change_flood = pd.read_csv('output/change_tyr_flood.csv')

###---01 plot for t-yr flood---###
##--Historical flood--##
flood_historical = flood[flood['model'].isin(['HBV True', 'HBV Recalib', 'Hymod', "LSTM"])]

fig, axs = plt.subplots(2,1,figsize=(6, 6), sharex=True)
sns.scatterplot(data=flood_historical, x='precip_rmse', y='20yr_flood', hue='model', alpha=0.6, ax=axs[0])
# Compute and plot LOWESS lines
for variable in flood_historical['model'].unique():
    subset = flood_historical[flood_historical['model'] == variable]
    lowess = sm.nonparametric.lowess(subset['20yr_flood'], subset['precip_rmse'], frac=0.5)  # Adjust frac as needed
    axs[0].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
axs[0].set_xlabel('Precipitation RMSE (mm/day)')
axs[0].set_ylabel('20-year flood (mm/day)')
axs[0].grid(True, linestyle ='--', alpha = 0.5)
axs[0].legend(title='', loc='best')

sns.scatterplot(data=flood_historical, x='precip_rmse', y='50yr_flood', hue='model', alpha=0.6, ax=axs[1])
# Compute and plot LOWESS lines
for variable in flood_historical['model'].unique():
    subset = flood_historical[flood_historical['model'] == variable]
    lowess = sm.nonparametric.lowess(subset['50yr_flood'], subset['precip_rmse'], frac=0.5)  # Adjust frac as needed
    axs[1].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
axs[1].set_xlabel('Precipitation RMSE (mm/day)')
axs[1].set_ylabel('50-year flood (mm/day)')
axs[1].grid(True, linestyle ='--', alpha = 0.5)
axs[1].get_legend().remove()
#save the plot
plt.savefig('output/tyr_flood_historical.png', dpi=300)
#clear the plot
plt.clf()


##--Future flood--##
flood_future = flood[flood['model'].isin([ 'HBV True Future', 'HBV Recalib Future', 'Hymod Future', "LSTM Future"])]
fig, axs = plt.subplots(2,1, figsize=(6,6), sharex=True)
sns.scatterplot(data=flood_future, x='precip_rmse', y='20yr_flood', hue='model', alpha=0.6, ax =axs[0])
for variable in flood_future['model'].unique():
    subset = flood_future[flood_future['model'] == variable]
    lowess =sm.nonparametric.lowess(subset['20yr_flood'], subset['precip_rmse'], frac=0.8)
    axs[0].plot(lowess[:,0], lowess[:,1], label=None, linewidth=3)
axs[0].set_xlabel('Precipitation RMSE (mm/day)')
axs[0].set_ylabel('20-year flood (mm/day)')
axs[0].grid(True, linestyle ='--', alpha = 0.5)
axs[0].get_legend().remove()

sns.scatterplot(data=flood_future, x='precip_rmse', y='50yr_flood', hue='model', alpha=0.6, ax=axs[1])
# Compute and plot LOWESS lines
for variable in flood_future['model'].unique():
    subset = flood_future[flood_future['model'] == variable]
    lowess = sm.nonparametric.lowess(subset['50yr_flood'], subset['precip_rmse'], frac=0.8)  # Adjust frac as needed
    axs[1].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
axs[1].set_xlabel('Precipitation RMSE (mm/day)')
axs[1].set_ylabel('50-year flood (mm/day)')
axs[1].grid(True, linestyle ='--', alpha = 0.5)
axs[1].legend(title='', loc='best')
#save the plot
plt.savefig('output/tyr_flood_future.png', dpi=300)
#clear the plot
plt.clf()


###---02 plot for change in t-yr flood---###
fig, axs = plt.subplots(2,1,figsize=(6, 6), sharex=True)
sns.scatterplot(data=change_flood, x='precip_rmse', y='change_20yr_flood', hue='model', alpha=0.6, ax=axs[0])
# Compute and plot LOWESS lines
for variable in change_flood['model'].unique():
    subset = change_flood[change_flood['model'] == variable]
    lowess = sm.nonparametric.lowess(subset['change_20yr_flood'], subset['precip_rmse'], frac=0.8)  # Adjust frac as needed
    axs[0].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
axs[0].set_xlabel('Precipitation RMSE (mm/day)')
axs[0].set_ylabel('∆20-year flood (mm/day)')
axs[0].grid(True, linestyle ='--', alpha = 0.5)
axs[0].legend(title='', loc='best')

sns.scatterplot(data=change_flood, x='precip_rmse', y='change_50yr_flood', hue='model', alpha=0.6, ax=axs[1])
# Compute and plot LOWESS lines
for variable in change_flood['model'].unique():
    subset = change_flood[change_flood['model'] == variable]
    lowess = sm.nonparametric.lowess(subset['change_50yr_flood'], subset['precip_rmse'], frac=0.8)  # Adjust frac as needed
    axs[1].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
axs[1].set_xlabel('Precipitation RMSE (mm/day)')
axs[1].set_ylabel('∆50-year flood (mm/day)')
axs[1].grid(True, linestyle ='--', alpha = 0.5)
axs[1].get_legend().remove()
#save the plot
plt.savefig('output/change_tyr_flood.png', dpi=300)
#clear the plot
plt.clf()