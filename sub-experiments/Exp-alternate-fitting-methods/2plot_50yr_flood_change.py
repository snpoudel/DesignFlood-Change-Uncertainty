#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#set colorblind friendly seaborn color
sns.set_palette('colorblind')

df = pd.read_csv('output/only50yr_flood.csv')
df = df[['model', 'method', 'distribution', 'precip_rmse', 'change_50yr_flood']]
df_zeroprecip = df[df['precip_rmse'] == 0]
df_zeroprecip['precip_cat'] = '0\n(n=30)'
df = df[df['precip_rmse'] != 0] #filter everything except precip zero
#convert precipitation error into categorical group
df['precip_cat']  = pd.cut(df['precip_rmse'], bins=[0,1,2,3,4,6,8],
                            labels=['0-1\n(n=21)', '1-2\n(n=49)', '2-3\n(n=66)', '3-4\n(n=65)', '4-6\n(n=82)', '6-8\n(n=34)'])
#merge back zero precips
df = pd.concat([df,df_zeroprecip], ignore_index=True)
df = df.dropna(axis='rows')

title_names = ['a) Gumbel distribution fitted with Maximum Likelihood', 'b) GEV distribution fitted with Maximum Likelihood', 'c) Gumbel distribution fitted with L-Moments', 'd) GEV distribution fitted with L-Moments']
#make a 2x2 plot for each method and distribution with precip error on x-axis and change in 50yr flood on y-axis
fig, ax = plt.subplots(2, 2, figsize=(10,6), sharex=True, sharey=True)
for i, method in enumerate(['mle', 'lm']):
    for j, distribution in enumerate(['gum', 'gev']):
        df_method = df[(df['method'] == method) & (df['distribution'] == distribution)]
        df_50yr = pd.melt(df_method, id_vars=['precip_cat', 'model'], value_vars=['change_50yr_flood'])
        df_50yr['objective'] = 'change_50yr'
        df_50yr = df_50yr.dropna(axis='rows')
        model_order = ['HBV Recalib', 'LSTM', 'FULL-HYMOD-LSTM', 'Full-Hymod', 'HYMOD-LSTM', 'Hymod']
        df_50yr['model'] = pd.Categorical(df_50yr['model'], categories=model_order, ordered=True)
        precip_cat_order = ['0\n(n=30)', '0-1\n(n=21)', '1-2\n(n=49)', '2-3\n(n=66)', '3-4\n(n=65)', '4-6\n(n=82)', '6-8\n(n=34)']
        sns.boxplot(data=df_50yr, order = precip_cat_order, width=0.8, linewidth=0.5,showfliers=False,
                    x='precip_cat', y='value', hue='model', ax=ax[i,j])
        ax[i,j].axhline(0, color='red', linewidth=1, linestyle='--')
        ax[i,j].set_title(title_names[i*2 + j])
        ax[i,j].set_xlabel('Precipitation Error (RMSE, mm/day)')
        ax[i,j].set_ylabel('Error in Δ50-yr Flood (%)')
        ax[i,j].grid(True, linestyle='--', alpha=0.4)  # Add grid
        if (i == 0 and j == 0):
            handles, labels = ax[i,j].get_legend_handles_labels()
            ax[i,j].legend(handles, labels, loc='upper center', ncol=3, fontsize='small', frameon=False)
        else:
            ax[i,j].get_legend().remove()
plt.suptitle('Climate change scenario: Temp:+2 CC:7% Precip:10%')            
plt.tight_layout()
plt.savefig('output/figures/Allmethods_50difference_plots.jpg', dpi=300)
plt.show()
