import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datashader as ds
from datashader.mpl_ext import dsshow

def using_datashader(ax, x, y, vmax = 50, colorbar = False):

    df = pd.DataFrame(dict(x=x, y=y))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        vmin=0,
        vmax=vmax,
        norm="linear",
        aspect="auto",
        ax=ax,
    )
    # #make xticks and yticks same
    x_ticks = ax.get_xticks()
    ticks = [0, int((np.max(x_ticks))/2), int(np.max(x_ticks))]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if colorbar == True:
        plt.colorbar(dsartist)

#read truth
used_basin_list = ['01108000', '01109060', '01177000', '01104500']
for id in used_basin_list:
    #read precip
    hist_precip = pd.read_csv(f'output/hist_precip_by_buckets_{id}.csv')
    future_precip = pd.read_csv(f'output/future_precip_by_buckets_{id}.csv')

    #read flow
    hist_flow = pd.read_csv(f'output/hist_flow_by_buckets_{id}.csv')
    future_flow = pd.read_csv(f'output/future_flow_by_buckets_{id}.csv')



    ##--HISTORICAL--##
    fig,axs=plt.subplots(4,4, figsize=(7,5))
    #precip
    data=hist_precip[hist_precip['tag']=='0-2']
    using_datashader(axs[0][0], data['truth'], data['PRECIP'])
    axs[0][0].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[0][0].set_title('MAP RMSE: 0-2')
    axs[0][0].set_ylabel('IDW MAP')
    axs[0][0].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=hist_precip[hist_precip['tag']=='2-4']
    using_datashader(axs[0][1], data['truth'], data['PRECIP'])
    axs[0][1].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[0][1].set_title('MAP RMSE: 2-4')
    axs[0][1].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=hist_precip[hist_precip['tag']=='4-6']
    using_datashader(axs[0][2], data['truth'], data['PRECIP'])
    axs[0][2].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[0][2].set_title('MAP RMSE: 4-6')
    axs[0][2].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)


    data=hist_precip[hist_precip['tag']=='6-8']
    using_datashader(axs[0][3], data['truth'], data['PRECIP'], colorbar = True)
    axs[0][3].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[0][3].set_title('MAP RMSE: 6-8')
    axs[0][3].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)


    #hbv recalibrated
    data=hist_flow[(hist_flow['model']=='HBV Re') & (hist_flow['tag']=='0-2')]
    using_datashader(axs[1][0], data['truth'], data['streamflow'])
    axs[1][0].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[1][0].set_ylabel('HBV-Re flow')
    axs[1][0].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)


    data=hist_flow[(hist_flow['model']=='HBV Re') & (hist_flow['tag']=='2-4')]
    using_datashader(axs[1][1], data['truth'], data['streamflow'])
    axs[1][1].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[1][1].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=hist_flow[(hist_flow['model']=='HBV Re') & (hist_flow['tag']=='4-6')]
    using_datashader(axs[1][2], data['truth'], data['streamflow'])
    axs[1][2].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[1][2].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=hist_flow[(hist_flow['model']=='HBV Re') & (hist_flow['tag']=='6-8')]
    using_datashader(axs[1][3], data['truth'], data['streamflow'] ,colorbar = True)
    axs[1][3].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[1][3].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    #HYMOD
    data=hist_flow[(hist_flow['model']=='HYMOD') & (hist_flow['tag']=='0-2')]
    using_datashader(axs[2][0], data['truth'], data['streamflow'])
    axs[2][0].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[2][0].set_ylabel('HYMOD flow')
    axs[2][0].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=hist_flow[(hist_flow['model']=='HYMOD') & (hist_flow['tag']=='2-4')]
    using_datashader(axs[2][1], data['truth'], data['streamflow'])
    axs[2][1].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[2][1].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=hist_flow[(hist_flow['model']=='HYMOD') & (hist_flow['tag']=='4-6')]
    using_datashader(axs[2][2], data['truth'], data['streamflow'])
    axs[2][2].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[2][2].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=hist_flow[(hist_flow['model']=='HYMOD') & (hist_flow['tag']=='4-6')]
    using_datashader(axs[2][3], data['truth'], data['streamflow'],colorbar = True)
    axs[2][3].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[2][3].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    #LSTM
    data=hist_flow[(hist_flow['model']=='LSTM') & (hist_flow['tag']=='0-2')]
    using_datashader(axs[3][0], data['truth'], data['streamflow'])
    axs[3][0].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[3][0].set_ylabel('LSTM flow')
    axs[3][0].set_xlabel('Truth')
    axs[3][0].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=hist_flow[(hist_flow['model']=='LSTM') & (hist_flow['tag']=='2-4')]
    using_datashader(axs[3][1], data['truth'], data['streamflow'])
    axs[3][1].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[3][1].set_xlabel('Truth')
    axs[3][1].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=hist_flow[(hist_flow['model']=='LSTM') & (hist_flow['tag']=='4-6')]
    using_datashader(axs[3][2], data['truth'], data['streamflow'])
    axs[3][2].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[3][2].set_xlabel('Truth')
    axs[3][2].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=hist_flow[(hist_flow['model']=='LSTM') & (hist_flow['tag']=='6-8')]
    using_datashader(axs[3][3], data['truth'], data['streamflow'],colorbar = True)
    axs[3][3].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[3][3].set_xlabel('Truth')
    axs[3][3].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    plt.tight_layout()
    plt.show()
    fig.savefig(f'output/figures/{id}/2historical_everything_qqplot.png', dpi=300)






    ##--Future--##
    fig,axs=plt.subplots(4,4, figsize=(7,5))
    #precip
    data=future_precip[future_precip['tag']=='0-2']
    using_datashader(axs[0][0], data['truth'], data['PRECIP'])
    axs[0][0].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[0][0].set_title('MAP RMSE: 0-2')
    axs[0][0].set_ylabel('IDW MAP')
    axs[0][0].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=future_precip[future_precip['tag']=='2-4']
    using_datashader(axs[0][1], data['truth'], data['PRECIP'])
    axs[0][1].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[0][1].set_title('MAP RMSE: 2-4')
    axs[0][1].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=future_precip[future_precip['tag']=='4-6']
    using_datashader(axs[0][2], data['truth'], data['PRECIP'])
    axs[0][2].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[0][2].set_title('MAP RMSE: 4-6')
    axs[0][2].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)


    data=future_precip[future_precip['tag']=='6-8']
    using_datashader(axs[0][3], data['truth'], data['PRECIP'], colorbar = True)
    axs[0][3].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[0][3].set_title('MAP RMSE: 6-8')
    axs[0][3].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)


    #hbv recalibrated
    data=future_flow[(future_flow['model']=='HBV Re') & (future_flow['tag']=='0-2')]
    using_datashader(axs[1][0], data['truth'], data['streamflow'])
    axs[1][0].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[1][0].set_ylabel('HBV-Re flow')
    axs[1][0].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)


    data=future_flow[(future_flow['model']=='HBV Re') & (future_flow['tag']=='2-4')]
    using_datashader(axs[1][1], data['truth'], data['streamflow'])
    axs[1][1].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[1][1].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=future_flow[(future_flow['model']=='HBV Re') & (future_flow['tag']=='4-6')]
    using_datashader(axs[1][2], data['truth'], data['streamflow'])
    axs[1][2].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[1][2].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=future_flow[(future_flow['model']=='HBV Re') & (future_flow['tag']=='6-8')]
    using_datashader(axs[1][3], data['truth'], data['streamflow'] ,colorbar = True)
    axs[1][3].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[1][3].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    #HYMOD
    data=future_flow[(future_flow['model']=='HYMOD') & (future_flow['tag']=='0-2')]
    using_datashader(axs[2][0], data['truth'], data['streamflow'])
    axs[2][0].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[2][0].set_ylabel('HYMOD flow')
    axs[2][0].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=future_flow[(future_flow['model']=='HYMOD') & (future_flow['tag']=='2-4')]
    using_datashader(axs[2][1], data['truth'], data['streamflow'])
    axs[2][1].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[2][1].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=future_flow[(future_flow['model']=='HYMOD') & (future_flow['tag']=='4-6')]
    using_datashader(axs[2][2], data['truth'], data['streamflow'])
    axs[2][2].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[2][2].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=future_flow[(future_flow['model']=='HYMOD') & (future_flow['tag']=='4-6')]
    using_datashader(axs[2][3], data['truth'], data['streamflow'],colorbar = True)
    axs[2][3].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[2][3].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    #LSTM
    data=future_flow[(future_flow['model']=='LSTM') & (future_flow['tag']=='0-2')]
    using_datashader(axs[3][0], data['truth'], data['streamflow'])
    axs[3][0].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[3][0].set_ylabel('LSTM flow')
    axs[3][0].set_xlabel('Truth')
    axs[3][0].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=future_flow[(future_flow['model']=='LSTM') & (future_flow['tag']=='2-4')]
    using_datashader(axs[3][1], data['truth'], data['streamflow'])
    axs[3][1].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[3][1].set_xlabel('Truth')
    axs[3][1].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=future_flow[(future_flow['model']=='LSTM') & (future_flow['tag']=='4-6')]
    using_datashader(axs[3][2], data['truth'], data['streamflow'])
    axs[3][2].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[3][2].set_xlabel('Truth')
    axs[3][2].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    data=future_flow[(future_flow['model']=='LSTM') & (future_flow['tag']=='6-8')]
    using_datashader(axs[3][3], data['truth'], data['streamflow'],colorbar = True)
    axs[3][3].axline(xy1=(0,0), slope=1, color='red', linewidth=0.3)
    axs[3][3].set_xlabel('Truth')
    axs[3][3].grid(True, linestyle='--',alpha = 0.1, linewidth = 0.3)

    plt.tight_layout()
    plt.show()
    fig.savefig(f'output/figures/{id}/3future_everything_qqplot.png', dpi=300)