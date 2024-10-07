#load library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#read dignostic csv file
# id = '01108000'
used_basin_list = ['01108000', '01109060', '01177000', '01104500']
for id in used_basin_list:
    df = pd.read_csv(f'output/diagnostics_validperiod_{id}.csv')
    if id == '01108000':
        df = df[df['grid'].isin([1,3,5,7,9])]
    else:
        df = df[df['grid'].isin([1,2,3,4,5])]
    #Plot the results
    #first melt the desired columns for boxplot
    df_bias_melt = pd.melt(df, id_vars=['grid'], value_vars=['BIAS(HBV)', 'BIAS(RECAL_HBV)', 'BIAS(HYMOD)', 'BIAS(LSTM)'])
    df_bias_melt['objective'] = 'BIAS'

    df_hfb_melt = pd.melt(df, id_vars=['grid'], value_vars= ['HFB(HBV)', 'HFB(RECAL_HBV)','HFB(HYMOD)' , 'HFB(LSTM)'])
    df_hfb_melt['objective'] = 'HFB'

    df_rmse_melt = pd.melt(df, id_vars=['grid'], value_vars=['RMSE(HBV)', 'RMSE(RECAL_HBV)','RMSE(HYMOD)' , 'RMSE(LSTM)'])
    df_rmse_melt['objective'] = 'RMSE'

    df_nse_melt = pd.melt(df, id_vars=['grid'], value_vars=['NSE(HBV)', 'NSE(RECAL_HBV)','NSE(HYMOD)' , 'NSE(LSTM)'])
    df_nse_melt['objective'] = 'NSE'

    df_kge_melt = pd.melt(df, id_vars=['grid'], value_vars=['KGE(HBV)','KGE(RECAL_HBV)', 'KGE(HYMOD)' , 'KGE(LSTM)'])
    df_kge_melt['objective'] = 'KGE'
    #combine all dataframes
    df_all = pd.concat([df_bias_melt, df_hfb_melt, df_rmse_melt, df_nse_melt, df_kge_melt], axis=0)
    df_all['model'] = df_all['variable'].apply(lambda x: x.split('(')[1].split(')')[0])
    #Make boxplots using seaborn
    seaplot = sns.catplot(
                data=df_all, x='grid', y='value', row='objective',
                hue='model', kind='box', showfliers = False,
                sharey=False,  legend_out=True,
                height = 2, aspect = 4, #aspect times height gives width of each facet
                ) 
    seaplot.set_axis_labels('Number gauging stations used', "") #set x and y labels
    seaplot.legend.set_title("Model") #set legend title
    seaplot.set_titles("") #remove default titles
    for index, ax in enumerate(seaplot.axes.flat): #seaplot.axes.flat is a list of all axes in facetgrid/catplot
        ax.set_ylabel(['Bias(%)', '>99.9th Flow Bias(%)', 'RMSE(mm/day)', 'NSE', 'KGE'][index])
        ax.grid(True, linestyle ='--', alpha = 0.5)
    plt.show()

    #save the plot
    seaplot.savefig(f'output/figures/{id}/4diagnostics_boxplot.png', dpi=300)

