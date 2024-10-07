#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read input
df = pd.read_csv('output/allbasins_diagnostics_entire_hist_future.csv')
df_zeroprecip = df[df['RMSE(PRECIP)'] == 0] #filter precip zero
df_zeroprecip['precip_cat'] = '0'

df = df[df['RMSE(PRECIP)'] != 0] #filter everything except precip zero
#convert precipitation error into categorical group
df['precip_cat']  = pd.cut(df['RMSE(PRECIP)'], bins=[0,1,2,3,4,6,8],
                           labels=['0-1', '1-2', '2-3', '3-4', '4-6', '6-8'])

#merge back zero precips
df = pd.concat([df,df_zeroprecip], ignore_index=True)
df = df.dropna(axis='rows')

# df = df[df['time']=='future']

#Filter basins that have all precip_error categories
# Group by 'precip_cat' and count unique 'station_id' for each precip error category
# station_counts = df.groupby('station_id')['precip_cat'].nunique()
# stations_with_all_errors = station_counts[station_counts == 7].index #7 precip categories
# df = df[df['station_id'].isin(stations_with_all_errors)]

#boxplot for no error

df_hfb_melt = pd.melt(df, id_vars=['precip_cat', 'time'], value_vars= ['HFB(HBV)', 'HFB(RECAL_HBV)','HFB(HYMOD)' , 'HFB(LSTM)'])
df_hfb_melt['objective'] = 'HFB'

#combine all dataframes
df_all = pd.concat([ df_hfb_melt], axis=0)
df_all['model'] = df_all['variable'].apply(lambda x: x.split('(')[1].split(')')[0])

#remove HBV model
df_all = df_all[df_all['model'] != 'HBV']


df_all_bias = df_all
#Make boxplots using seaborn
precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
seaplot = sns.catplot(
            data=df_all_bias, order = precip_cat_order,
            x='precip_cat', y='value', row='time',
            hue='model', kind='box', showfliers = False, width =0.5,
            sharey=False,  legend_out=True,
            height = 3, aspect = 3, #aspect times height gives width of each facet
            ) 
seaplot.set_axis_labels('Average Precipitation Uncertainty (mm/day)', "") #set x and y labels
seaplot.legend.set_title("Model") #set legend title
seaplot.set_titles("") #remove default titles
for index, ax in enumerate(seaplot.axes.flat): #seaplot.axes.flat is a list of all axes in facetgrid/catplot
    ax.set_ylabel(['HFB(Historical)', 'HFB(Future)'][index])
    # ax.set_ylabel([ 'HFB(Future)'][index])
    ax.grid(True, linestyle ='--', alpha = 0.5)
# plt.tight_layout()
plt.show()

#save the plot
seaplot.savefig('output/figures/hist_future_diagnosticsHFB_allbasin.png', dpi=300)