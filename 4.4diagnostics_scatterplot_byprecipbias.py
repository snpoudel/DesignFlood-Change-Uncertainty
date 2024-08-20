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

# 1. NSE
nse_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
             value_vars=['NSE(HBV)', 'NSE(RECAL_HBV)','NSE(HYMOD)', 'NSE(LSTM)'])

plt1 = sns.lmplot(data=nse_melt, x='RMSE(PRECIP)', y='value', hue='variable',
            lowess = True, scatter_kws={'alpha':.6}, line_kws={'alpha':1, 'lw':4},
            height=5, aspect=1.2)
plt1.set_xlabels('Precipitation RMSE (mm/day)')
plt1.set_ylabels('Streamflow NSE')
plt1.ax.grid(True, linestyle ='--', alpha = 0.5)
plt1._legend.set_bbox_to_anchor([0.3, 0.25])
plt1._legend.set_title('')
new_labels = ['HBV', 'RECAL_HBV', 'HYMOD', 'LSTM']
for t, l in zip(plt1._legend.texts, new_labels): t.set_text(l)
#save the plot
plt1.savefig('output/scatterplot_nse.png', dpi=300)

# 2. KGE
kge_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
             value_vars=['KGE(HBV)', 'KGE(RECAL_HBV)', 'KGE(HYMOD)', 'KGE(LSTM)'])

plt2 = sns.lmplot(data=kge_melt, x='RMSE(PRECIP)', y='value', hue='variable',
            lowess = True, scatter_kws={'alpha':.6}, line_kws={'alpha':1, 'lw':4},
            height=5, aspect=1.2)
plt2.set_xlabels('Precipitation RMSE (mm/day)')
plt2.set_ylabels('Streamflow KGE')
plt2.ax.grid(True, linestyle ='--', alpha = 0.5)
plt2._legend.set_bbox_to_anchor([0.3, 0.25])
plt2._legend.set_title('')
new_labels = ['HBV', 'RECAL_HBV', 'HYMOD', 'LSTM']
for t, l in zip(plt2._legend.texts, new_labels): t.set_text(l)
#save the plot
plt2.savefig('output/scatterplot_kge.png', dpi=300)

# 3. BIAS
bias_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
             value_vars=['BIAS(HBV)', 'BIAS(RECAL_HBV)', 'BIAS(HYMOD)', 'BIAS(LSTM)'])

plt3 = sns.lmplot(data=bias_melt, x='RMSE(PRECIP)', y='value', hue='variable',
            lowess = True, scatter_kws={'alpha':.6}, line_kws={'alpha':1, 'lw':4},
            height=5, aspect=1.2)
plt3.set_xlabels('Precipitation RMSE (mm/day)')
plt3.set_ylabels('Streamflow Bias')
plt3.ax.grid(True, linestyle ='--', alpha = 0.5)
plt3._legend.set_bbox_to_anchor([0.3, 0.8])
plt3._legend.set_title('')
new_labels = ['HBV', 'RECAL_HBV', 'HYMOD', 'LSTM']
for t, l in zip(plt3._legend.texts, new_labels): t.set_text(l)
#save the plot
plt3.savefig('output/scatterplot_bias.png', dpi=300)

# 4. HFB
hfb_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
             value_vars=['HFB(HBV)', 'HFB(RECAL_HBV)', 'HFB(HYMOD)', 'HFB(LSTM)'])

plt4 = sns.lmplot(data=hfb_melt, x='RMSE(PRECIP)', y='value', hue='variable',
            lowess = True, scatter_kws={'alpha':.6}, line_kws={'alpha':1, 'lw':4},
            height=5, aspect=1.2)
plt4.set_xlabels('Precipitation RMSE (mm/day)')
plt4.set_ylabels('>99.9th Streamflow Bias')
plt4.ax.grid(True, linestyle ='--', alpha = 0.5)
plt4._legend.set_bbox_to_anchor([0.3, 0.75])
plt4._legend.set_title('')
new_labels = ['HBV', 'RECAL_HBV', 'HYMOD', 'LSTM']
for t, l in zip(plt4._legend.texts, new_labels): t.set_text(l)
#save the plot
plt4.savefig('output/scatterplot_hfb.png', dpi=300)

# 5. RMSE
rmse_melt = pd.melt(df, id_vars=['RMSE(PRECIP)'],
             value_vars=['RMSE(HBV)', 'RMSE(RECAL_HBV)', 'RMSE(HYMOD)', 'RMSE(LSTM)'])

plt5 = sns.lmplot(data=rmse_melt, x='RMSE(PRECIP)', y='value', hue='variable',
            lowess = True, scatter_kws={'alpha':.6}, line_kws={'alpha':1, 'lw':4},
            height=5, aspect=1.2)
plt5.set_xlabels('Precipitation RMSE (mm/day)')
plt5.set_ylabels('Streamflow RMSE')
plt5.ax.grid(True, linestyle ='--', alpha = 0.5)
plt5._legend.set_bbox_to_anchor([0.3, 0.75])
plt5._legend.set_title('')
new_labels = ['HBV', 'RECAL_HBV', 'HYMOD', 'LSTM']
for t, l in zip(plt5._legend.texts, new_labels): t.set_text(l)
#save the plot
plt5.savefig('output/scatterplot_rmse.png', dpi=300)