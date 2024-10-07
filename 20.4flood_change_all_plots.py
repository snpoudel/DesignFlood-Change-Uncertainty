#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read input
df = pd.read_csv('output/allbasins_change_tyr_flood.csv')
df_zeroprecip = df[df['precip_rmse'] == 0] #filter precip zero
df_zeroprecip['precip_cat'] = '0'

#find median values of change in flood for true model
df_zerotrue = df_zeroprecip[df_zeroprecip['model'] == 'HBV True']
med_5yr, med10yr, med20yr = np.median(df_zerotrue['change_5yr_flood']), np.median(df_zerotrue['change_10yr_flood']), np.median(df_zerotrue['change_20yr_flood'])
median_list = [med_5yr, med10yr, med20yr]

df = df[df['precip_rmse'] != 0] #filter everything except precip zero
#convert precipitation error into categorical group
df['precip_cat']  = pd.cut(df['precip_rmse'], bins=[0,1,2,3,4,6,8],
                           labels=['0-1', '1-2', '2-3', '3-4', '4-6', '6-8'])
df = df[df['model'] != 'HBV True']

#merge back zero precips
df = pd.concat([df,df_zeroprecip], ignore_index=True)
df = df.dropna(axis='rows')

#Filter basins that have all precip_error categories
# Group by 'precip_cat' and count unique 'station_id' for each precip error categor
# station_counts = df.groupby('station')['precip_cat'].nunique()
# stations_with_all_errors = station_counts[station_counts == 7].index #7 precip categories
# df = df[df['station'].isin(stations_with_all_errors)]

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
#remove HBV model


#Make boxplots using seaborn
precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
seaplot = sns.catplot(
            data=df_all, order = precip_cat_order,
            x='precip_cat', y='value', row='objective',
            hue='model', kind='box', showfliers = False, width = 0.5,
            sharey=False,  legend_out=True,
            height = 3, aspect = 3, #aspect times height gives width of each facet
            ) 
seaplot.set_axis_labels('Average Precipitation Uncertainty (mm/day)', "") #set x and y labels
seaplot.legend.set_title("Model") #set legend title
seaplot.set_titles("") #remove default titles
for index, ax in enumerate(seaplot.axes.flat): #seaplot.axes.flat is a list of all axes in facetgrid/catplot
    ax.set_ylabel(['Δ in 5yr-flood (%)', 'Δ in 10yr-flood (%)', 'Δ in 20yr-flood (%)'][index])
    ax.axhline(y=median_list[index], color = 'red', linestyle='--')
    ax.grid(True, linestyle ='--', alpha = 0.5)
plt.show()
#save the plot
seaplot.savefig('output/figures/tyr-flood_allbasin.png', dpi=300)