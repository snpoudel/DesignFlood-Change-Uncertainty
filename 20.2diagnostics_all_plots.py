#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#set colorblind friendly colors from seaborn
sns.set_palette('colorblind')

#read input
df = pd.read_csv('output/allbasins_diagnostics_validperiod.csv')
df_zeroprecip = df[df['RMSE(PRECIP)'] == 0] #filter precip zero
df_zeroprecip['precip_cat'] = '0(n=30)'

df = df[df['RMSE(PRECIP)'] != 0] #filter everything except precip zero
#convert precipitation error into categorical group
df['precip_cat']  = pd.cut(df['RMSE(PRECIP)'], bins=[0,1,2,3,4,6,8],
                           labels=['0-1(n=21)', '1-2(n=49)', '2-3(n=66)', '3-4(n=65)', '4-6(n=82)', '6-8(n=34)'])

#merge back zero precips
df = pd.concat([df,df_zeroprecip], ignore_index=True)
df = df.dropna(axis='rows')

#Filter basins that have all precip_error categories
# Group by 'precip_cat' and count unique 'station_id' for each precip error category
# station_counts = df.groupby('station_id')['precip_cat'].nunique()
# stations_with_all_errors = station_counts[station_counts == 7].index #7 precip categories
# df = df[df['station_id'].isin(stations_with_all_errors)]

#boxplot for no error
df_bias_melt = pd.melt(df, id_vars=['precip_cat'], value_vars=['BIAS(HBV)', 'BIAS(RECAL_HBV)',  'BIAS(FULL-HYMOD)','BIAS(LSTM)', 'BIAS(HYMOD-LSTM)','BIAS(HYMOD)', 'BIAS(FULL-HYMOD-LSTM)'])
df_bias_melt['objective'] = 'BIAS'

df_hfb_melt = pd.melt(df, id_vars=['precip_cat'], value_vars= ['HFB(HBV)', 'HFB(RECAL_HBV)', 'HFB(FULL-HYMOD)', 'HFB(LSTM)', 'HFB(HYMOD-LSTM)','HFB(HYMOD)' , 'HFB(FULL-HYMOD-LSTM)'])
df_hfb_melt['objective'] = 'HFB'

df_rmse_melt = pd.melt(df, id_vars=['precip_cat'], value_vars=['RMSE(HBV)', 'RMSE(RECAL_HBV)', 'RMSE(FULL-HYMOD)', 'RMSE(LSTM)', 'RMSE(HYMOD-LSTM)','RMSE(HYMOD)' , 'RMSE(FULL-HYMOD-LSTM)'])
df_rmse_melt['objective'] = 'RMSE'

df_nse_melt = pd.melt(df, id_vars=['precip_cat'], value_vars=['NSE(HBV)', 'NSE(RECAL_HBV)', 'NSE(FULL-HYMOD)', 'NSE(LSTM)', 'NSE(HYMOD-LSTM)','NSE(HYMOD)' , 'NSE(FULL-HYMOD-LSTM)'])
df_nse_melt['objective'] = 'NSE'

df_kge_melt = pd.melt(df, id_vars=['precip_cat'], value_vars=['KGE(HBV)','KGE(RECAL_HBV)',  'KGE(FULL-HYMOD)', 'KGE(LSTM)', 'KGE(HYMOD-LSTM)','KGE(HYMOD)' , 'KGE(FULL-HYMOD-LSTM)'])
df_kge_melt['objective'] = 'KGE'
#combine all dataframes
df_all = pd.concat([df_bias_melt, df_hfb_melt, df_rmse_melt, df_nse_melt, df_kge_melt], axis=0)
df_all['model'] = df_all['variable'].apply(lambda x: x.split('(')[1].split(')')[0])

#remove HBV model
df_all = df_all[df_all['model'] != 'HBV']
#change name of model RECAL_HBV to HBV-Recalib
df_all['model'] = df_all['model'].replace('RECAL_HBV', 'HBV-Recalib')

#set order of model categories
model_order = ['HBV-Recalib', 'LSTM', 'FULL-HYMOD-LSTM', 'FULL-HYMOD', 'HYMOD-LSTM',  'HYMOD']

df_all['model'] = pd.Categorical(df_all['model'], categories=model_order, ordered=True)

#make plot for nse, kge, rmse
df_all_nkr = df_all[df_all['objective'].isin(['RMSE','KGE','NSE'])]
#Make boxplots using seaborn
# precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
precip_cat_order = ['0(n=30)', '0-1(n=21)', '1-2(n=49)', '2-3(n=66)', '3-4(n=65)', '4-6(n=82)', '6-8(n=34)']
seaplot = sns.catplot(
            data=df_all_nkr, order = precip_cat_order,
            x='precip_cat', y='value', row='objective',
            hue='model', kind='box', showfliers = False, width =0.8,
            sharey=False,  legend_out=True,
            height = 2.5, aspect = 2.5, #aspect times height gives width of each facet
            ) 
seaplot.set_axis_labels('Precipitation Uncertainty (RMSE, mm/day)', "") #set x and y labels
seaplot.legend.set_title("Model") #set legend title
seaplot.set_titles("") #remove default titles
plt.suptitle('Summary performance during validation period for all basins')
for index, ax in enumerate(seaplot.axes.flat): #seaplot.axes.flat is a list of all axes in facetgrid/catplot
    ax.set_ylabel(['RMSE(mm/day)', 'NSE', 'KGE'][index])
    ax.grid(True, linestyle ='--', alpha = 0.5)
# plt.tight_layout()
plt.show()
#save the plot
seaplot.savefig('output/figures/NSE_allbasin.png', dpi=300)
#also save as a pdf
# seaplot.savefig('output/figures/diagnosticsNSE_allbasin.pdf')
# seaplot.savefig('output/figures/NoLSTM_diagnosticsKGE_allbasin.png', dpi=300) #without lstm
#make plot for nse, kge, rmse
df_all_bias = df_all[df_all['objective'].isin(['BIAS','HFB'])]
#Make boxplots using seaborn
# precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
seaplot = sns.catplot(
            data=df_all_bias, order = precip_cat_order,
            x='precip_cat', y='value', row='objective',
            hue='model', kind='box', showfliers = False, width =0.7,
            sharey=False,  legend_out=True,
            height = 3, aspect = 3, #aspect times height gives width of each facet
            ) 
seaplot.set_axis_labels('Precipitation Uncertainty (RMSE, mm/day)', "") #set x and y labels
seaplot.legend.set_title("Model") #set legend title
seaplot.set_titles("") #remove default titles
plt.suptitle('Summary performance during validation period for all basins')
for index, ax in enumerate(seaplot.axes.flat): #seaplot.axes.flat is a list of all axes in facetgrid/catplot
    ax.set_ylabel(['Bias(%)', '>99.9th Flow Bias(%)'][index])
    ax.grid(True, linestyle ='--', alpha = 0.5)
# plt.tight_layout()
plt.show()

#save the plot
# seaplot.savefig('output/figures/NoLSTM_diagnosticsBIAS_allbasin.png', dpi=300) #without lstm
seaplot.savefig('output/figures/BIAS_allbasin.png', dpi=300)


#######################################################################################################################################################################################################
#only save result for nse plot
df_all_nkr = df_all[df_all['objective'].isin(['NSE'])]
#Make boxplots using seaborn
# precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
seaplot = sns.catplot(
            data=df_all_nkr, order = precip_cat_order,
            x='precip_cat', y='value', row='objective',
            hue='model', kind='box', showfliers = False, width =0.8,
            sharey=False,  legend_out=False,
            height = 4, aspect = 2, #aspect times height gives width of each facet
            ) 
seaplot.set_axis_labels('Precipitation Uncertainty (RMSE, mm/day)', "") #set x and y labels
seaplot.legend.set_title("Model") #set legend title
seaplot.set_titles("") #remove default titles
plt.ylabel('Nash-Sutcliffe Efficiency')
plt.grid(True, linestyle ='--', alpha = 0.5)
plt.ylim(0, None)
plt.tight_layout()
plt.show()
#save the plot
seaplot.savefig('output/figures/OnlyNSE_allbasin.jpg', dpi=300)

#######################################################################################################################################################################################################
#barplot with median value and error bar as 25th and 75th percentile
df_all_nkr = df_all[df_all['objective'].isin(['NSE'])]
#make a custom function to calculate median and 25th and 75th percentile
def median_25_75(x):
    return pd.Series([np.percentile(x, 25), np.percentile(x, 75)])
#Make boxplots using seaborn
# precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
seaplot = sns.catplot(
            data=df_all_nkr, order = precip_cat_order,
            x='precip_cat', y='value', row='objective',
            hue='model', kind='bar', estimator=np.median, width = 0.8,
            errorbar=median_25_75, capsize=0.2, errwidth=1.3,
            sharey=False,  legend_out=False,
            height = 4, aspect = 2, #aspect times height gives width of each facet
            edgecolor='black', linewidth = 0.5 # set edge color of box as black
            ) 
seaplot.set_axis_labels('Precipitation Uncertainty (RMSE, mm/day)', "") #set x and y labels
seaplot.legend.set_title("Model") #set legend title
seaplot.set_titles("") #remove default titles
plt.ylabel('Nash-Sutcliffe Efficiency (NSE)')
#set horizontal grid
for ax in seaplot.axes.flat:
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, None)
plt.tight_layout()
seaplot._legend.set_bbox_to_anchor((0.28, 0.52))
seaplot._legend.get_frame().set_alpha(0.90)
plt.show()
#save the plot
seaplot.savefig('output/figures/OnlyNSE_barplot_allbasin.jpg', dpi=300)
seaplot.savefig('FINAL-FIGURES/4NSE_barplot_allbasin.jpg', dpi=300)



#######################################################################################################################################################################################################
#only show nse plot for model_order = ['RECAL_HBV', 'FULL-HYMOD', 'HYMOD']
df_all_process = df_all[df_all['model'].isin(['HBV-Recalib', 'FULL-HYMOD', 'HYMOD'])]
model_order = ['HBV-Recalib', 'FULL-HYMOD', 'HYMOD']
df_all_process['model'] = pd.Categorical(df_all_process['model'], categories=model_order, ordered=True)
#make plot for nse, kge, rmse
df_all_nkr = df_all_process[df_all_process['objective'].isin(['NSE'])]
#Make boxplots using seaborn
# precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
seaplot = sns.catplot(
            data=df_all_nkr, order = precip_cat_order,
            x='precip_cat', y='value', row='objective',
            hue='model', kind='box', showfliers = False, width =0.8,
            sharey=False,  legend_out=False,
            height = 4, aspect = 2, #aspect times height gives width of each facet
            )
seaplot.set_axis_labels('Precipitation Uncertainty (RMSE, mm/day)', "") #set x and y labels
seaplot.legend.set_title("Model") #set legend title
seaplot.set_titles("") #remove default titles
plt.ylabel('Nash-Sutcliffe Efficiency')
plt.grid(True, linestyle ='--', alpha = 0.5)
plt.ylim(0, None)
plt.tight_layout()
plt.show()
#save the plot
seaplot.savefig('output/figures/OnlyNSE_ProcessModels_allbasin.png', dpi=300)


#######################################################################################################################################################################################################
#only show nse plot for model_order = ['FULL-HYMOD-LSTM', 'HYMOD-LSTM', 'LSTM']
df_all = df_all[df_all['model'].isin(['FULL-HYMOD-LSTM', 'HYMOD-LSTM', 'LSTM'])]
model_order = ['FULL-HYMOD-LSTM', 'HYMOD-LSTM', 'LSTM']
df_all['model'] = pd.Categorical(df_all['model'], categories=model_order, ordered=True)
#make plot for nse, kge, rmse
df_all_nkr = df_all[df_all['objective'].isin(['NSE'])]
#Make boxplots using seaborn
# precip_cat_order = ['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8']
seaplot = sns.catplot(
            data=df_all_nkr, order = precip_cat_order,
            x='precip_cat', y='value', row='objective',
            hue='model', kind='box', showfliers = False, width =0.8,
            sharey=False,  legend_out=False,
            height = 4, aspect = 2, #aspect times height gives width of each facet
            )
seaplot.set_axis_labels('Precipitation Uncertainty (RMSE, mm/day)', "") #set x and y labels
seaplot.legend.set_title("Model") #set legend title
seaplot.set_titles("") #remove default titles
plt.ylabel('Nash-Sutcliffe Efficiency')
plt.grid(True, linestyle ='--', alpha = 0.5)
plt.ylim(0, None)
plt.tight_layout()
plt.show()
#save the plot
seaplot.savefig('output/figures/OnlyNSE_LSTMModels_allbasin.png', dpi=300)