#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#set colorblind friendly seaborn color
sns.set_palette('colorblind')

data3 = pd.read_csv('output/flood_change_scenario3.csv')
data3 = data3[['model', 'change_50yr_flood', 'precip_rmse']]
data3_zeroprecip = data3[data3['precip_rmse'] == 0]
data3_zeroprecip['precip_cat'] = '0\n(n=30)'
data3 = data3[data3['precip_rmse'] != 0]
data3['precip_cat'] = pd.cut(data3['precip_rmse'], bins=[0,1,2,3,4,6,8],
                            labels=['0-1\n(n=21)', '1-2\n(n=49)', '2-3\n(n=66)', '3-4\n(n=65)', '4-6\n(n=82)', '6-8\n(n=34)'])
data3 = pd.concat([data3, data3_zeroprecip], ignore_index=True)
data3 = data3.dropna(axis='rows')
data3['scenario'] = 'scenario3'


data7 = pd.read_csv('output/flood_change_scenario7.csv')
data7 = data7[['model', 'change_50yr_flood', 'precip_rmse']]
data7_zeroprecip = data7[data7['precip_rmse'] == 0]
data7_zeroprecip['precip_cat'] = '0\n(n=30)'
data7 = data7[data7['precip_rmse'] != 0]
data7['precip_cat'] = pd.cut(data7['precip_rmse'], bins=[0,1,2,3,4,6,8],
                            labels=['0-1\n(n=21)', '1-2\n(n=49)', '2-3\n(n=66)', '3-4\n(n=65)', '4-6\n(n=82)', '6-8\n(n=34)'])
data7 = pd.concat([data7, data7_zeroprecip], ignore_index=True)
data7 = data7.dropna(axis='rows')
data7['scenario'] = 'scenario7'


data11 = pd.read_csv('output/flood_change_scenario11.csv')
data11 = data11[['model', 'change_50yr_flood', 'precip_rmse']]
data11_zeroprecip = data11[data11['precip_rmse'] == 0]
data11_zeroprecip['precip_cat'] = '0\n(n=30)'
data11 = data11[data11['precip_rmse'] != 0]
data11['precip_cat'] = pd.cut(data11['precip_rmse'], bins=[0,1,2,3,4,6,8],
                            labels=['0-1\n(n=21)', '1-2\n(n=49)', '2-3\n(n=66)', '3-4\n(n=65)', '4-6\n(n=82)', '6-8\n(n=34)'])
data11 = pd.concat([data11, data11_zeroprecip], ignore_index=True)
data11 = data11.dropna(axis='rows')
data11['scenario'] = 'scenario11'


data15 = pd.read_csv('output/flood_change_scenario15.csv')
data15 = data15[['model', 'change_50yr_flood', 'precip_rmse']]
data15_zeroprecip = data15[data15['precip_rmse'] == 0]
data15_zeroprecip['precip_cat'] = '0\n(n=30)'
data15 = data15[data15['precip_rmse'] != 0]
data15['precip_cat'] = pd.cut(data15['precip_rmse'], bins=[0,1,2,3,4,6,8],
                            labels=['0-1\n(n=21)', '1-2\n(n=49)', '2-3\n(n=66)', '3-4\n(n=65)', '4-6\n(n=82)', '6-8\n(n=34)'])
data15 = pd.concat([data15, data15_zeroprecip], ignore_index=True)
data15 = data15.dropna(axis='rows')
data15['scenario'] = 'scenario15'

#order precipitation categories
data3['precip_cat'] = pd.Categorical(data3['precip_cat'], categories=['0\n(n=30)', '0-1\n(n=21)', '1-2\n(n=49)', '2-3\n(n=66)', '3-4\n(n=65)', '4-6\n(n=82)', '6-8\n(n=34)'], ordered=True)
data7['precip_cat'] = pd.Categorical(data7['precip_cat'], categories=['0\n(n=30)', '0-1\n(n=21)', '1-2\n(n=49)', '2-3\n(n=66)', '3-4\n(n=65)', '4-6\n(n=82)', '6-8\n(n=34)'], ordered=True)
data11['precip_cat'] = pd.Categorical(data11['precip_cat'], categories=['0\n(n=30)', '0-1\n(n=21)', '1-2\n(n=49)', '2-3\n(n=66)', '3-4\n(n=65)', '4-6\n(n=82)', '6-8\n(n=34)'], ordered=True)
data15['precip_cat'] = pd.Categorical(data15['precip_cat'], categories=['0\n(n=30)', '0-1\n(n=21)', '1-2\n(n=49)', '2-3\n(n=66)', '3-4\n(n=65)', '4-6\n(n=82)', '6-8\n(n=34)'], ordered=True)

#order model categories
data3['model'] = data3['model'].replace({'Gr4j Recalib':'GR4J-Recalib', 'HYMOD-LSTM':'HYMOD(PP)', 'Hymod':'HYMOD'})
data7['model'] = data7['model'].replace({'Gr4j Recalib':'GR4J-Recalib', 'HYMOD-LSTM':'HYMOD(PP)', 'Hymod':'HYMOD'})
data11['model'] = data11['model'].replace({'Gr4j Recalib':'GR4J-Recalib', 'HYMOD-LSTM':'HYMOD(PP)', 'Hymod':'HYMOD'})
data15['model'] = data15['model'].replace({'Gr4j Recalib':'GR4J-Recalib', 'HYMOD-LSTM':'HYMOD(PP)', 'Hymod':'HYMOD'})

# #order model categories
model_order = ['GR4J-Recalib', 'LSTM', 'HYMOD(PP)', 'HYMOD']
data3['model'] = pd.Categorical(data3['model'], categories=model_order, ordered=True)
data7['model'] = pd.Categorical(data7['model'], categories=model_order, ordered=True)
data11['model'] = pd.Categorical(data11['model'], categories=model_order, ordered=True)
data15['model'] = pd.Categorical(data15['model'], categories=model_order, ordered=True)

title_names = ['a) Temp:+2 CC:3.5% Precip:0%', 'b) Temp:+2 CC:3.5% Precip:10%', 'c) Temp:+2 CC:7% Precip:0%', 'd) Temp:+2 CC:7% Precip:10%']

#make a 2x2 plot for each scenario with precip error on x-axis and change in 10yr flood on y-axis
fig, ax = plt.subplots(2, 2, figsize=(10,6), sharex=True, sharey=True)
for i, (scenario, title) in enumerate(zip([data3, data7, data11, data15], title_names)):
    sns.boxplot(x='precip_cat', y='change_50yr_flood', hue='model', width=0.8, showfliers=False, data=scenario, ax=ax[i//2, i%2], linewidth=0.5)
    ax[i//2, i%2].set_title(title)
    ax[i//2, i%2].set_xlabel('Precipitation Error (RMSE, mm/day)')
    ax[i//2, i%2].set_ylabel('Error in Δ50-yr Flood (%)')
    ax[i//2, i%2].grid(True, linestyle ='--', alpha=0.4)  # Add grid
    ax[i//2, i%2].axhline(0, color='red', linestyle='-.', linewidth=1)  # Add horizontal line at 0
    if i == 0 :  # Show legend only in the first and third subplot
        ax[i//2, i%2].legend(loc='upper center', ncol=2, fontsize='small')
    else:
        ax[i//2, i%2].legend_.remove()
# plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)  # Reduce gaps between subplots
plt.tight_layout()
plt.savefig('output/figures/50yr_flood_change.jpg', dpi=300)
plt.show()
