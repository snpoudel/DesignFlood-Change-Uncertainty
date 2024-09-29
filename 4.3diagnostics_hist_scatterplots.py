#load library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

#basin id
used_basin_list = ['01108000', '01109060', '01177000', '01104500']
for id in used_basin_list:
    # id = '01109060'
    #read dignostic csv file
    df = pd.read_csv(f'output/diagnostics_validperiod_{id}.csv')

    #make a scatter plot of grid vs RMSE(PRECIP)
    df_without99 = df[~(df['grid']==99)]
    plt.figure(figsize=(6,4))
    plt.scatter(df_without99['grid'], df_without99['RMSE(PRECIP)'], color='blue', alpha=0.6)
    plt.xlabel('Number of gauging stations used')
    plt.ylabel('Precipitation RMSE (mm/day)')
    #change x ticks labels
    plt.grid(True, linestyle ='--', alpha = 0.5)
    plt.tight_layout()
    plt.savefig(f'output/figures/{id}/precip rmse vs num of stations used.png', dpi=300)
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
    fig,axs = plt.subplots(3,1,figsize=(6, 7),sharex=True)
    sns.scatterplot(data=nse_melt, x='RMSE(PRECIP)', y='value', hue='variable', alpha=0.6, ax=axs[0])
    # Compute and plot LOWESS lines
    for variable in nse_melt['variable'].unique():
        subset = nse_melt[nse_melt['variable'] == variable]
        lowess = sm.nonparametric.lowess(subset['value'], subset['RMSE(PRECIP)'], frac=0.5)  # Adjust frac as needed
        axs[0].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=1.5)
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
        lowess = sm.nonparametric.lowess(subset['value'], subset['RMSE(PRECIP)'], frac=0.5)  # Adjust frac as needed
        axs[1].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=1.5)
    # Customize the plot
    axs[1].set_xlabel('Precipitation RMSE (mm/day)')
    axs[1].set_ylabel('Streamflow KGE')
    axs[1].grid(True, linestyle ='--', alpha = 0.5)
    axs[1].get_legend().remove()

    sns.scatterplot(data=rmse_melt, x='RMSE(PRECIP)', y='value', hue='variable', alpha=0.6,ax=axs[2])
    # Compute and plot LOWESS lines
    for variable in rmse_melt['variable'].unique():
        subset = rmse_melt[rmse_melt['variable'] == variable]
        lowess = sm.nonparametric.lowess(subset['value'], subset['RMSE(PRECIP)'], frac=0.5)  # Adjust frac as needed
        axs[2].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=1.5)
    # Customize the plot
    axs[2].set_xlabel('Precipitation RMSE (mm/day)')
    axs[2].set_ylabel('Streamflow RMSE')
    axs[2].grid(True, linestyle ='--', alpha = 0.5)
    axs[2].get_legend().remove()
    #save the plot
    plt.tight_layout()
    plt.savefig(f'output/figures/{id}/scatterplot_nse_kge_rmse.png', dpi=300)
    # Clear the plot
    plt.clf()  # Clear the current figure



    # 2. BIAS, HFB
    bias_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
                value_vars=['BIAS(HBV)', 'BIAS(RECAL_HBV)', 'BIAS(HYMOD)', 'BIAS(LSTM)'])
    hfb_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
                value_vars=['HFB(HBV)', 'HFB(RECAL_HBV)', 'HFB(HYMOD)', 'HFB(LSTM)'])

    fig,axs = plt.subplots(2,1, figsize=(6, 7), sharex=True)
    #Plot scatter points
    sns.scatterplot(data=bias_melt, x='RMSE(PRECIP)', y='value', hue='variable', alpha=0.5, ax=axs[0])
    # Compute and plot LOWESS lines
    for variable in bias_melt['variable'].unique():
        subset = bias_melt[bias_melt['variable'] == variable]
        lowess = sm.nonparametric.lowess(subset['value'], subset['RMSE(PRECIP)'], frac=0.5)  # Adjust frac as needed
        axs[0].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=1.5)
    # Customize the plot
    axs[0].set_xlabel('Precipitation RMSE (mm/day)')
    axs[0].set_ylabel('Streamflow Bias (%)\n(obs - sim) / obs')
    axs[0].grid(True, linestyle ='--', alpha = 0.5)
    new_labels = ['HBV', 'RECALIBRATED HBV', 'HYMOD', 'LSTM']
    for t, l in zip(axs[0].get_legend().texts, new_labels):t.set_text(l)
    sns.scatterplot(data=hfb_melt, x='RMSE(PRECIP)', y='value', hue='variable', alpha=0.6, ax=axs[1])
    # Compute and plot LOWESS lines
    for variable in hfb_melt['variable'].unique():
        subset = hfb_melt[hfb_melt['variable'] == variable]
        lowess = sm.nonparametric.lowess(subset['value'], subset['RMSE(PRECIP)'], frac=0.5)  # Adjust frac as needed
        axs[1].plot(lowess[:, 0], lowess[:, 1], label=None, linewidth=1.5)
    # Customize the plot
    axs[1].set_xlabel('Precipitation RMSE (mm/day)')
    axs[1].set_ylabel('99.9th Streamflow HFB (%)')
    axs[1].grid(True, linestyle ='--', alpha = 0.5)
    axs[1].get_legend().remove()
    plt.tight_layout()
    #save the plot
    plt.savefig(f'output/figures/{id}/scatterplot_bias_hfb.png', dpi=300)
    # Clear the plot
    plt.clf()  # Clear the current figure
    plt.close() # Close the current figure window