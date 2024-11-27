#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#set colorblind friendly seaborn color
sns.set_palette('colorblind')

#loop through all combinations of method and distribution and make boxplots
for method in ['mle', 'lm']:
    for distribution in ['gev', 'gum']:
        #filter precip zero
        df_zeroprecip = df[df['precip_rmse'] == 0] 
        df_zeroprecip['precip_cat'] = '0'

        df = df[df['precip_rmse'] != 0] #filter everything except precip zero
        #convert precipitation error into categorical group
        df['precip_cat']  = pd.cut(df['precip_rmse'], bins=[0,1,2,3,4,6,8],
                                labels=['0-1', '1-2', '2-3', '3-4', '4-6', '6-8'])

        #merge back zero precips
        df = pd.concat([df,df_zeroprecip], ignore_index=True)
        df = df.dropna(axis='rows')
        #only keep df with this method and distribution
        df = df[(df['method'] == method) & (df['distribution'] == distribution)]

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
        # df_all = df_all[df_all['model'] != 'LSTM'] #remove LSTM model as well

        # #set order of model categories
        model_order = ['HBV Recalib', 'FULL-HYMOD-LSTM', 'Full-Hymod', 'LSTM', 'HYMOD-LSTM', 'Hymod']
        df_all['model'] = pd.Categorical(df_all['model'], categories=model_order, ordered=True)

        #Make boxplots using seaborn
        precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
        seaplot = sns.catplot(
                    data=df_all, order = precip_cat_order,
                    x='precip_cat', y='value', row='objective',
                    hue='model', kind='box', showfliers = False, width = 0.5,
                    sharey=False,  legend_out=True,
                    height = 3, aspect = 3, #aspect times height gives width of each facet
                    ) 
        seaplot.set_axis_labels('Precipitation Uncertainty (RMSE mm/day)', "") #set x and y labels
        seaplot.legend.set_title("Model") #set legend title
        seaplot.set_titles("") #remove default titles
        for index, ax in enumerate(seaplot.axes.flat): #seaplot.axes.flat is a list of all axes in facetgrid/catplot
            ax.set_ylabel(['Δ in 5yr-flood (%)', 'Δ in 10yr-flood (%)\n(Est Δ - True Δ) / True Δ', 'Δ in 20yr-flood (%)'][index])
            ax.axhline(y=0, linestyle='--', color='red', alpha=0.5)
            ax.grid(True, linestyle ='--', alpha = 0.5)
        plt.show()
        #save the plot
        seaplot.savefig(f'output/figures/change_flood_{distribution}_{method}.png', dpi=300)
        # seaplot.savefig('output/figures/NoLSTM_tyr-flood_allbasin.png', dpi=300)