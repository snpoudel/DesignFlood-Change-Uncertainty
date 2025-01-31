#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#set colorblind friendly seaborn color
sns.set_palette('colorblind')
df = pd.read_csv('output/allbasins_pooling_tyr_flood_modified.csv')

distribution = 'gev'
#loop through all combinations of method and distribution and make boxplots
for method in ['mle']:
    # for distribution in ['gev']:
    #filter precip zero
    df_zeroprecip = df[df['precip_rmse'] == 0] 
    df_zeroprecip['precip_cat'] = '0'

    df = df[df['precip_rmse'] != 0] #filter everything except precip zero
    #convert precipitation error into categorical group
    df['precip_cat']  = pd.cut(df['precip_rmse'], bins=[0,1,2,3,4,6,8],
                            labels=['0-1', '1-2', '2-3', '3-4', '4-6', '6-8'])

    #merge back zero precips
    df = pd.concat([df,df_zeroprecip], ignore_index=True)
    df = df.drop(columns=['change_5yr_flood', 'change_10yr_flood', 'change_20yr_flood'])
    df = df.dropna(axis='rows')
    #only keep df with this method and distribution
    df = df[(df['method'] == method) & (df['distribution'] == distribution)]
    #apply pooling by calculating average est_change in flood across all basins for a precip category
    df['avg_est_change_5yr_flood'] = df.groupby(['precip_cat', 'model'])['est_change_5yr_flood'].transform('mean')
    df['avg_est_change_10yr_flood'] = df.groupby(['precip_cat', 'model'])['est_change_10yr_flood'].transform('mean')
    df['avg_est_change_20yr_flood'] = df.groupby(['precip_cat', 'model'])['est_change_20yr_flood'].transform('mean')

    #calculate change in tyr flood
    df['change_5yr_flood'] = df['avg_est_change_5yr_flood'] - df['true_change_5yr_flood']
    df['change_10yr_flood'] = df['avg_est_change_10yr_flood'] - df['true_change_10yr_flood']
    df['change_20yr_flood'] = df['avg_est_change_20yr_flood'] - df['true_change_20yr_flood']

    #deletelater#
    #only keep df for HBV recalib at precip error 0
    new_df = df[(df['model'] == 'HBV Recalib') & (df['precip_cat'] == '0-1')]
    #only keep column: station, model, precip_cat, true_change_5yr_flood, est_change_5yr_flood, avg_est_change_5yr_flood, change_5yr_flood
    new_df = new_df[['station', 'model', 'precip_cat', 'true_change_5yr_flood', 'est_change_5yr_flood', 'avg_est_change_5yr_flood', 'change_5yr_flood']]
    #only keep rows with unique station
    new_df = new_df.drop_duplicates(subset='station')
    #delete later#

    #boxplot for no error
    df_5yr = pd.melt(df, id_vars=['precip_cat', 'model'], value_vars=['change_5yr_flood'])
    df_5yr['objective'] = 'change_5yr'

    df_10yr = pd.melt(df, id_vars=['precip_cat', 'model'], value_vars=['change_10yr_flood'])
    df_10yr['objective'] = 'change_10yr'

    df_20yr = pd.melt(df, id_vars=['precip_cat', 'model'], value_vars=['change_20yr_flood'])
    df_20yr['objective'] = 'change_20yr'

    #combine all dataframes
    df_all = pd.concat([df_5yr, df_10yr, df_20yr], axis=0)
    df_all = df_all.dropna(axis='rows')

    df_all = df_all[df_all['model'] != 'LSTM'] #remove LSTM model as well
    df_all = df_all[df_all['model'] != 'FULL-HYMOD-LSTM'] 
    df_all = df_all[df_all['model'] != 'HYMOD-LSTM']

    # #set order of model categories
    # model_order = ['HBV Recalib', 'FULL-HYMOD-LSTM', 'Full-Hymod', 'LSTM', 'HYMOD-LSTM', 'Hymod']
    model_order = ['HBV Recalib', 'Full-Hymod', 'Hymod']
    df_all['model'] = pd.Categorical(df_all['model'], categories=model_order, ordered=True)

    #Make boxplots using seaborn
    precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
    seaplot = sns.catplot(
                data=df_all, order = precip_cat_order,
                x='precip_cat', y='value', row='objective',
                hue='model', kind='box', showfliers = False, width = 0.8,
                sharey=False,  legend_out=True,
                height = 2.5, aspect = 2.5, #aspect times height gives width of each facet
                ) 
    seaplot.set_axis_labels('Precipitation Uncertainty (RMSE mm/day)', "") #set x and y labels
    seaplot.legend.set_title("Model") #set legend title
    seaplot.set_titles("") #remove default titles
    for index, ax in enumerate(seaplot.axes.flat): #seaplot.axes.flat is a list of all axes in facetgrid/catplot
        ax.set_ylabel(['Δ in 25yr-flood (%)', 'Δ in 50yr-flood (%)\n Δ Model(%) - Δ True(%)', 'Δ in 100yr-flood (%)'][index])
        ax.axhline(y=0, linestyle='--', color='red', alpha=0.5)
        ax.grid(True, linestyle ='--', alpha = 0.5)
    plt.show()
    #save the plot
    seaplot.savefig(f'output/figures/pooled_difference_flood_{distribution}_{method}.png', dpi=300)
    #also save as a pdf file
    # seaplot.savefig(f'output/figures/change_flood_{distribution}_{method}.pdf')
    # seaplot.savefig('output/figures/NoLSTM_tyr-flood_allbasin.png', dpi=300)