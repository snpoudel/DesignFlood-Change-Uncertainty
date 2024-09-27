import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# id = '01109060'
used_basin_list = ['01108000', '01109060', '01177000', '01104500']
for id in used_basin_list:
    # Load the data
    flood = pd.read_csv(f'output/tyr_flood_{id}.csv')
    change_flood = pd.read_csv(f'output/change_tyr_flood_{id}.csv')

    ###---01 plot for t-yr flood---###
    ##--Historical flood--##
    flood_historical = flood[flood['model'].isin(['HBV True', 'HBV Recalib', 'Hymod', "LSTM"])]
    flood_historical = flood_historical.reset_index(drop=True)

    fig, axs = plt.subplots(3,1,figsize=(7, 8), sharex=True)
    sns.scatterplot(data=flood_historical, x='precip_rmse', y='5yr_flood', hue='model', alpha=0.6, ax=axs[0])
    # Compute and plot LOWESS lines
    for variable in flood_historical['model'].unique():
        subset = flood_historical[flood_historical['model'] == variable]
        lowess = sm.nonparametric.lowess(subset['5yr_flood'], subset['precip_rmse'], frac=0.5)  # Adjust frac as needed
        axs[0].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
    axs[0].set_xlabel('Precipitation RMSE (mm/day)')
    axs[0].set_ylabel('5-year flood (mm/day)')
    axs[0].grid(True, linestyle ='--', alpha = 0.5)
    axs[0].get_legend().remove()
    axs[0].set_title('Flood return period with historical precip')
    sns.scatterplot(data=flood_historical, x='precip_rmse', y='10yr_flood', hue='model', alpha=0.6, ax=axs[1])
    # Compute and plot LOWESS lines
    for variable in flood_historical['model'].unique():
        subset = flood_historical[flood_historical['model'] == variable]
        lowess = sm.nonparametric.lowess(subset['10yr_flood'], subset['precip_rmse'], frac=0.5)  # Adjust frac as needed
        axs[1].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
    axs[1].set_xlabel('Precipitation RMSE (mm/day)')
    axs[1].set_ylabel('10-year flood (mm/day)')
    axs[1].grid(True, linestyle ='--', alpha = 0.5)
    axs[1].get_legend().remove()

    sns.scatterplot(data=flood_historical, x='precip_rmse', y='20yr_flood', hue='model', alpha=0.6, ax=axs[2])
    # Compute and plot LOWESS lines
    for variable in flood_historical['model'].unique():
        subset = flood_historical[flood_historical['model'] == variable]
        lowess = sm.nonparametric.lowess(subset['20yr_flood'], subset['precip_rmse'], frac=0.5)  # Adjust frac as needed
        axs[2].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
    axs[2].set_xlabel('Precipitation RMSE (mm/day)')
    axs[2].set_ylabel('20-year flood (mm/day)')
    axs[2].grid(True, linestyle ='--', alpha = 0.5)
    axs[2].legend(title='', loc='best')
    #save the plot
    plt.tight_layout()
    plt.savefig(f'output/figures/{id}/tyr_flood_historical.png', dpi=300)
    #clear the plot
    plt.clf()


    ##--Future flood--##
    flood_future = flood[flood['model'].isin([ 'HBV True Future', 'HBV Recalib Future', 'Hymod Future', "LSTM Future"])]
    fig, axs = plt.subplots(3,1, figsize=(7,8), sharex=True)
    sns.scatterplot(data=flood_future, x='precip_rmse', y='5yr_flood', hue='model', alpha=0.6, ax =axs[0])
    for variable in flood_future['model'].unique():
        subset = flood_future[flood_future['model'] == variable]
        lowess =sm.nonparametric.lowess(subset['5yr_flood'], subset['precip_rmse'], frac=0.8)
        axs[0].plot(lowess[:,0], lowess[:,1], label=None, linewidth=3)
    axs[0].set_xlabel('Precipitation RMSE (mm/day)')
    axs[0].set_ylabel('5-year flood (mm/day)')
    axs[0].grid(True, linestyle ='--', alpha = 0.5)
    axs[0].get_legend().remove()
    axs[0].set_title('Flood return period with future precip')

    sns.scatterplot(data=flood_future, x='precip_rmse', y='10yr_flood', hue='model', alpha=0.6, ax=axs[1])
    # Compute and plot LOWESS lines
    for variable in flood_future['model'].unique():
        subset = flood_future[flood_future['model'] == variable]
        lowess = sm.nonparametric.lowess(subset['10yr_flood'], subset['precip_rmse'], frac=0.8)  # Adjust frac as needed
        axs[1].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
    axs[1].set_xlabel('Precipitation RMSE (mm/day)')
    axs[1].set_ylabel('10-year flood (mm/day)')
    axs[1].grid(True, linestyle ='--', alpha = 0.5)
    axs[1].get_legend().remove()

    sns.scatterplot(data=flood_future, x='precip_rmse', y='20yr_flood', hue='model', alpha=0.6, ax=axs[2])
    # Compute and plot LOWESS lines
    for variable in flood_future['model'].unique():
        subset = flood_future[flood_future['model'] == variable]
        lowess = sm.nonparametric.lowess(subset['20yr_flood'], subset['precip_rmse'], frac=0.8)  # Adjust frac as needed
        axs[2].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
    axs[2].set_xlabel('Precipitation RMSE (mm/day)')
    axs[2].set_ylabel('20-year flood (mm/day)')
    axs[2].grid(True, linestyle ='--', alpha = 0.5)
    axs[2].legend(title='', loc='best')
    #save the plot
    plt.tight_layout()
    plt.savefig(f'output/figures/{id}/tyr_flood_future.png', dpi=300)
    #clear the plot
    plt.clf()


    ###---02 plot for change in t-yr flood---###
    fig, axs = plt.subplots(3,1,figsize=(7, 8), sharex=True)
    sns.scatterplot(data=change_flood, x='precip_rmse', y='change_5yr_flood', hue='model', alpha=0.6, ax=axs[0])
    # Compute and plot LOWESS lines
    for variable in change_flood['model'].unique():
        subset = change_flood[change_flood['model'] == variable]
        lowess = sm.nonparametric.lowess(subset['change_5yr_flood'], subset['precip_rmse'], frac=0.8)  # Adjust frac as needed
        axs[0].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
    axs[0].set_xlabel('Precipitation RMSE (mm/day)')
    axs[0].set_ylabel('∆5-year flood (mm/day)')
    axs[0].grid(True, linestyle ='--', alpha = 0.5)
    axs[0].get_legend().remove()
    axs[0].set_title('Change in flood return period')

    sns.scatterplot(data=change_flood, x='precip_rmse', y='change_10yr_flood', hue='model', alpha=0.6, ax=axs[1])
    # Compute and plot LOWESS lines
    for variable in change_flood['model'].unique():
        subset = change_flood[change_flood['model'] == variable]
        lowess = sm.nonparametric.lowess(subset['change_10yr_flood'], subset['precip_rmse'], frac=0.8)  # Adjust frac as needed
        axs[1].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
    axs[1].set_xlabel('Precipitation RMSE (mm/day)')
    axs[1].set_ylabel('∆10-year flood (mm/day)')
    axs[1].grid(True, linestyle ='--', alpha = 0.5)
    axs[1].get_legend().remove()

    sns.scatterplot(data=change_flood, x='precip_rmse', y='change_20yr_flood', hue='model', alpha=0.6, ax=axs[2])
    # Compute and plot LOWESS lines
    for variable in change_flood['model'].unique():
        subset = change_flood[change_flood['model'] == variable]
        lowess = sm.nonparametric.lowess(subset['change_20yr_flood'], subset['precip_rmse'], frac=0.8)  # Adjust frac as needed
        axs[2].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=3)
    axs[2].set_xlabel('Precipitation RMSE (mm/day)')
    axs[2].set_ylabel('∆20-year flood (mm/day)')
    axs[2].grid(True, linestyle ='--', alpha = 0.5)
    axs[2].legend(title='', loc='best')
    #save the plot
    plt.tight_layout()
    plt.savefig(f'output/figures/{id}/change_tyr_flood.png', dpi=300)
    #clear the plot
    plt.clf()