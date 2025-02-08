import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

basin_list = pd.read_csv('data/ma29basins.csv', dtype={'basin_id':str})# used_basin_list = ['01108000', '01109060', '01177000', '01104500']
# basin_list = basin_list[basin_list['basin_id'].isin(used_basin_list)]
id = '01108000'
# for id in basin_list['basin_id']:

#case 1, precip error =0, all stations used
palette0 = [sns.color_palette('Set2')[0]]

hymod_params = pd.DataFrame()
grid = 99
for comb in np.arange(10):
    if os.path.exists(f'output/parameters/hymod/params{id}_grid{grid}_comb{comb}.csv'):
        file = pd.read_csv(f'output/parameters/hymod/params{id}_grid{grid}_comb{comb}.csv')
        file['tag'] = f'error0'
        hymod_params=pd.concat([hymod_params, file], ignore_index=True)
#merge truth
df_error0 = hymod_params

#case 2, using n-1 stations
palette1 = [sns.color_palette('Set2')[1]]
hymod_params = pd.DataFrame()
total_grid = basin_list[basin_list['basin_id'] == id]['num_stations'].values[0]
grid = total_grid - 2
for comb in np.arange(10):
    if os.path.exists(f'output/parameters/hymod/params{id}_grid{grid}_comb{comb}.csv'):
        file = pd.read_csv(f'output/parameters/hymod/params{id}_grid{grid}_comb{comb}.csv')
        file['tag'] = f'error1out'
        hymod_params=pd.concat([hymod_params, file], ignore_index=True)
#merge truth
df_error1out = hymod_params

#case 3, using n-2 stations
palette2 = [sns.color_palette('Set2')[2]]
hymod_params = pd.DataFrame()
total_grid = basin_list[basin_list['basin_id'] == id]['num_stations'].values[0]
grid = total_grid - 8
for comb in np.arange(10):
    if os.path.exists(f'output/parameters/hymod/params{id}_grid{grid}_comb{comb}.csv'):
        file = pd.read_csv(f'output/parameters/hymod/params{id}_grid{grid}_comb{comb}.csv')
        file['tag'] = f'error1out'
        hymod_params=pd.concat([hymod_params, file], ignore_index=True)
#merge truth
df_error2out = hymod_params

#only keep some interested parameters
filtered_params =['kpwp','coeff_pet','ks','ddf','ts','crf', 'tag']
df_error0 = df_error0.loc[:,filtered_params]
df_error1out = df_error1out.loc[:,filtered_params]
df_error2out = df_error2out.loc[:,filtered_params]


# vars=['kpwp','etexp','hmax','bexp','alpha','ks','lmax','coeff_pet','ddf',
#                             'scf','ts','tm','tti','whc','crf','maxbas']
# vars_limit=[[0.0001,0.999],[0.01,1.99],[5,1000],[0.01,1.99],[0.01,0.99],[0.0005,0.99],
#                             [1,2000], [0.5,2],[0.05,10],
#                             [0.5,2],[-1,4],[-1,4],[-1,4],[0,0.2],[0.1,1],[1,10]]
vars=['kpwp','coeff_pet','ks','ddf','ts','crf']
vars_limit=[[0.001,0.999],[0.5,2],[0.0005,0.99],[0.05,10],[-1,4],[0.1,1]]

# Set up the figure and axes
total_cases = 3 #total precip error cases
fig, axs = plt.subplots(nrows=1*total_cases, ncols=6, figsize=(9, 5))
axs = axs.flatten()

# Create a box plot for each variable
for i, var in enumerate(vars):
    plt.suptitle(f'HYMOD calibrated parameters for basin: {id} \na)true precip with n=10 gauges b) precip with n-2 gauges c) precip with n-4 gauges')
    sns.swarmplot(data=df_error0, y=var, ax=axs[i], hue='tag', palette=palette0, size =5)
    # axs[i].set_title(f'{vars_label[i]}')
    axs[i].set_title(vars[i])
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    axs[i].set_ylim(vars_limit[i])
    axs[i].grid(True, linestyle='--', alpha=0.3)
    axs[i].get_legend().remove()

# Create a box plot for each variable
for i, var in enumerate(vars):
    inew = i+6
    sns.swarmplot(data=df_error1out, y=var, ax=axs[inew], hue='tag', palette=palette1 , size =5)
    # axs[i].set_title(f'{vars_label[i]}')
    axs[inew].set_title(vars[i])
    axs[inew].set_xlabel('')
    axs[inew].set_ylabel('')
    axs[inew].set_ylim(vars_limit[i])
    axs[inew].grid(True, linestyle='--', alpha=0.3)
    axs[inew].get_legend().remove()

# Create a box plot for each variable
for i, var in enumerate(vars):
    inew = i+6+6
    sns.swarmplot(data=df_error2out, y=var, ax=axs[inew], hue='tag', palette=palette2 , size =5)
    # axs[i].set_title(f'{vars_label[i]}')
    axs[inew].set_title(vars[i])
    axs[inew].set_xlabel('')
    axs[inew].set_ylabel('')
    axs[inew].set_ylim(vars_limit[i])
    axs[inew].grid(True, linestyle='--', alpha=0.3)
    axs[inew].get_legend().remove()
# Turn off any unused subplots
# axs[17].remove()
# axs[35].remove()
# axs[53].remove()

# Adjust layout
plt.tight_layout()
plt.show()

#save plot
# fig.savefig(f'output/figures/{id}/10param_comparision.png', dpi=300)
fig.savefig(f'output/figures/HYMOD_calib_parameters.png', dpi=300)