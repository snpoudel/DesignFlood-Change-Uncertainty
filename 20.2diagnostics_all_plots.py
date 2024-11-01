#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read input
df = pd.read_csv('output/allbasins_diagnostics_validperiod.csv')
df_zeroprecip = df[df['RMSE(PRECIP)'] == 0] #filter precip zero
df_zeroprecip['precip_cat'] = '0'

df = df[df['RMSE(PRECIP)'] != 0] #filter everything except precip zero
#convert precipitation error into categorical group
df['precip_cat']  = pd.cut(df['RMSE(PRECIP)'], bins=[0,1,2,3,4,6,8],
                           labels=['0-1', '1-2', '2-3', '3-4', '4-6', '6-8'])

#merge back zero precips
df = pd.concat([df,df_zeroprecip], ignore_index=True)
df = df.dropna(axis='rows')

#Filter basins that have all precip_error categories
# Group by 'precip_cat' and count unique 'station_id' for each precip error category
# station_counts = df.groupby('station_id')['precip_cat'].nunique()
# stations_with_all_errors = station_counts[station_counts == 7].index #7 precip categories
# df = df[df['station_id'].isin(stations_with_all_errors)]

#boxplot for no error
df_bias_melt = pd.melt(df, id_vars=['precip_cat'], value_vars=['BIAS(HBV)', 'BIAS(RECAL_HBV)', 'BIAS(HYMOD)', 'BIAS(LSTM)'])
df_bias_melt['objective'] = 'BIAS'

df_hfb_melt = pd.melt(df, id_vars=['precip_cat'], value_vars= ['HFB(HBV)', 'HFB(RECAL_HBV)','HFB(HYMOD)' , 'HFB(LSTM)'])
df_hfb_melt['objective'] = 'HFB'

df_rmse_melt = pd.melt(df, id_vars=['precip_cat'], value_vars=['RMSE(HBV)', 'RMSE(RECAL_HBV)','RMSE(HYMOD)' , 'RMSE(LSTM)'])
df_rmse_melt['objective'] = 'RMSE'

df_nse_melt = pd.melt(df, id_vars=['precip_cat'], value_vars=['NSE(HBV)', 'NSE(RECAL_HBV)','NSE(HYMOD)' , 'NSE(LSTM)'])
df_nse_melt['objective'] = 'NSE'

df_kge_melt = pd.melt(df, id_vars=['precip_cat'], value_vars=['KGE(HBV)','KGE(RECAL_HBV)', 'KGE(HYMOD)' , 'KGE(LSTM)'])
df_kge_melt['objective'] = 'KGE'
#combine all dataframes
df_all = pd.concat([df_bias_melt, df_hfb_melt, df_rmse_melt, df_nse_melt, df_kge_melt], axis=0)
df_all['model'] = df_all['variable'].apply(lambda x: x.split('(')[1].split(')')[0])

#remove HBV model
df_all = df_all[df_all['model'] != 'HBV']
# df_all = df_all[df_all['model'] != 'LSTM'] #remove LSTM model as well

#make plot for nse, kge, rmse
df_all_nkr = df_all[df_all['objective'].isin(['RMSE','NSE','KGE'])]
#Make boxplots using seaborn
precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
seaplot = sns.catplot(
            data=df_all_nkr, order = precip_cat_order,
            x='precip_cat', y='value', row='objective',
            hue='model', kind='box', showfliers = False, width =0.5,
            sharey=False,  legend_out=True,
            height = 3, aspect = 3, #aspect times height gives width of each facet
            ) 
seaplot.set_axis_labels('Average Precipitation Uncertainty (mm/day)', "") #set x and y labels
seaplot.legend.set_title("Model") #set legend title
seaplot.set_titles("") #remove default titles
plt.suptitle('Diagnostics plot during validation period for all basins')
for index, ax in enumerate(seaplot.axes.flat): #seaplot.axes.flat is a list of all axes in facetgrid/catplot
    ax.set_ylabel(['RMSE(mm/day)', 'NSE', 'KGE'][index])
    ax.grid(True, linestyle ='--', alpha = 0.5)
# plt.tight_layout()
plt.show()
#save the plot
seaplot.savefig('output/figures/diagnosticsNSE_allbasin.png', dpi=300)
# seaplot.savefig('output/figures/NoLSTM_diagnosticsKGE_allbasin.png', dpi=300) #without lstm
#make plot for nse, kge, rmse
df_all_bias = df_all[df_all['objective'].isin(['BIAS','HFB'])]
#Make boxplots using seaborn
precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
seaplot = sns.catplot(
            data=df_all_bias, order = precip_cat_order,
            x='precip_cat', y='value', row='objective',
            hue='model', kind='box', showfliers = False, width =0.5,
            sharey=False,  legend_out=True,
            height = 3, aspect = 3, #aspect times height gives width of each facet
            ) 
seaplot.set_axis_labels('Average Precipitation Uncertainty (mm/day)', "") #set x and y labels
seaplot.legend.set_title("Model") #set legend title
seaplot.set_titles("") #remove default titles
plt.suptitle('Diagnostics plot during validation period for all basins')
for index, ax in enumerate(seaplot.axes.flat): #seaplot.axes.flat is a list of all axes in facetgrid/catplot
    ax.set_ylabel(['Bias(%)', '>99.9th Flow Bias(%)'][index])
    ax.grid(True, linestyle ='--', alpha = 0.5)
# plt.tight_layout()
plt.show()

#save the plot
# seaplot.savefig('output/figures/NoLSTM_diagnosticsBIAS_allbasin.png', dpi=300) #without lstm
seaplot.savefig('output/figures/diagnosticsBIAS_allbasin.png', dpi=300)