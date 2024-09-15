import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#color palette
palette = sns.color_palette('colorblind')
#read truth
id = '01109060'
#read precip
hist_precip = pd.read_csv(f'output/hist_precip_by_buckets_{id}.csv')
future_precip = pd.read_csv(f'output/future_precip_by_buckets_{id}.csv')

#read flow
hist_flow = pd.read_csv(f'output/hist_flow_by_buckets_{id}.csv')
future_flow = pd.read_csv(f'output/future_flow_by_buckets_{id}.csv')


##--HISTORICAL--##
fig,axs=plt.subplots(4,4, figsize=(9,8))
#precip
axs[0][0].scatter(data=hist_precip[hist_precip['tag']=='0-2'],
                  x='truth',y='PRECIP',alpha=0.2,color=palette[0])
axs[0][0].axline(xy1=(0,0), slope=1, color='#555555')
axs[0][0].set_title('MAP RMSE: 0-2')
axs[0][0].set_ylabel('Interpolated MAP')
axs[0][0].set_xlim(-10,180)
axs[0][0].set_ylim(-10,180)
# axs[0][0].set_aspect('equal')
axs[0][0].grid(True, linestyle='--',alpha = 0.5)

axs[0][1].scatter(data=hist_precip[hist_precip['tag']=='2-4'],
                  x='truth',y='PRECIP',alpha=0.2,color=palette[0])
axs[0][1].axline(xy1=(0,0), slope=1, color='#555555')
axs[0][1].set_title('MAP RMSE: 2-4')
axs[0][1].set_xlim(-10,180)
axs[0][1].set_ylim(-10,180)
# axs[0][1].set_aspect('equal')
axs[0][1].grid(True, linestyle='--',alpha = 0.5)

axs[0][2].scatter(data=hist_precip[hist_precip['tag']=='4-6'],
                  x='truth',y='PRECIP',alpha=0.2,color=palette[0])
axs[0][2].axline(xy1=(0,0), slope=1, color='#555555')
axs[0][2].set_title('MAP RMSE: 4-6')
axs[0][2].set_xlim(-10,180)
axs[0][2].set_ylim(-10,180)
# axs[0][2].set_aspect('equal')
axs[0][2].grid(True, linestyle='--',alpha = 0.5)


axs[0][3].scatter(data=hist_precip[hist_precip['tag']=='6-8'],
                  x='truth',y='PRECIP',alpha=0.2,color=palette[0])
axs[0][3].axline(xy1=(0,0), slope=1, color='#555555')
axs[0][3].set_title('MAP RMSE: 6-8')
axs[0][3].set_aspect('equal')
axs[0][3].set_xlim(-10,180)
axs[0][3].set_ylim(-10,180)
# axs[0][3].set_aspect('equal')
axs[0][3].grid(True, linestyle='--',alpha = 0.5)


#hbv recalibrated
axs[1][0].scatter(data=hist_flow[(hist_flow['model']=='HBV Re') & (hist_flow['tag']=='0-2')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[1])
axs[1][0].axline(xy1=(0,0), slope=1, color='#555555')
axs[1][0].set_ylabel('Simulated HBV-Re flow')
axs[1][0].set_xlim(-1.2,25)
axs[1][0].set_ylim(-1.2,25)
# axs[1][0].set_aspect('equal')
axs[1][0].grid(True, linestyle='--',alpha = 0.5)


axs[1][1].scatter(data=hist_flow[(hist_flow['model']=='HBV Re') & (hist_flow['tag']=='2-4')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[1])
axs[1][1].axline(xy1=(0,0), slope=1, color='#555555')
axs[1][1].set_xlim(-1.2,25)
axs[1][1].set_ylim(-1.2,25)
# axs[1][1].set_aspect('equal')
axs[1][1].grid(True, linestyle='--',alpha = 0.5)

axs[1][2].scatter(data=hist_flow[(hist_flow['model']=='HBV Re') & (hist_flow['tag']=='4-6')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[1])
axs[1][2].axline(xy1=(0,0), slope=1, color='#555555')
axs[1][2].set_xlim(-1.2,25)
axs[1][2].set_ylim(-1.2,25)
# axs[1][2].set_aspect('equal')
axs[1][2].grid(True, linestyle='--',alpha = 0.5)

axs[1][3].scatter(data=hist_flow[(hist_flow['model']=='HBV Re') & (hist_flow['tag']=='6-8')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[1])
axs[1][3].axline(xy1=(0,0), slope=1, color='#555555')
axs[1][3].set_xlim(-1.2,25)
axs[1][3].set_ylim(-1.2,25)
# axs[1][3].set_aspect('equal')
axs[1][3].grid(True, linestyle='--',alpha = 0.5)

#HYMOD
axs[2][0].scatter(data=hist_flow[(hist_flow['model']=='HYMOD') & (hist_flow['tag']=='0-2')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[2])
axs[2][0].axline(xy1=(0,0), slope=1, color='#555555')
axs[2][0].set_ylabel('Simulated HYMOD flow')
axs[2][0].set_xlim(-1.2,25)
axs[2][0].set_ylim(-1.2,25)
# axs[2][0].set_aspect('equal')
axs[2][0].grid(True, linestyle='--',alpha = 0.5)

axs[2][1].scatter(data=hist_flow[(hist_flow['model']=='HYMOD') & (hist_flow['tag']=='2-4')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[2])
axs[2][1].axline(xy1=(0,0), slope=1, color='#555555')
axs[2][1].set_xlim(-1.2,25)
axs[2][1].set_ylim(-1.2,25)
# axs[2][1].set_aspect('equal')
axs[2][1].grid(True, linestyle='--',alpha = 0.5)

axs[2][2].scatter(data=hist_flow[(hist_flow['model']=='HYMOD') & (hist_flow['tag']=='4-6')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[2])
axs[2][2].axline(xy1=(0,0), slope=1, color='#555555')
axs[2][2].set_xlim(-1.2,25)
axs[2][2].set_ylim(-1.2,25)
# axs[2][2].set_aspect('equal')
axs[2][2].grid(True, linestyle='--',alpha = 0.5)

axs[2][3].scatter(data=hist_flow[(hist_flow['model']=='HYMOD') & (hist_flow['tag']=='6-8')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[2])
axs[2][3].axline(xy1=(0,0), slope=1, color='#555555')
axs[2][3].set_xlim(-1.2,25)
axs[2][3].set_ylim(-1.2,25)
# axs[2][3].set_aspect('equal')
axs[2][3].grid(True, linestyle='--',alpha = 0.5)

#LSTM
axs[3][0].scatter(data=hist_flow[(hist_flow['model']=='LSTM') & (hist_flow['tag']=='0-2')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[3])
axs[3][0].axline(xy1=(0,0), slope=1, color='#555555')
axs[3][0].set_ylabel('Simulated LSTM flow')
axs[3][0].set_xlabel('Truth')
axs[3][0].set_xlim(-1.2,25)
axs[3][0].set_ylim(-1.2,25)
# axs[3][0].set_aspect('equal')
axs[3][0].grid(True, linestyle='--',alpha = 0.5)

axs[3][1].scatter(data=hist_flow[(hist_flow['model']=='LSTM') & (hist_flow['tag']=='2-4')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[3])
axs[3][1].axline(xy1=(0,0), slope=1, color='#555555')
axs[3][1].set_xlabel('Truth')
axs[3][1].set_xlim(-1.2,25)
axs[3][1].set_ylim(-1.2,25)
# axs[3][1].set_aspect('equal')
axs[3][1].grid(True, linestyle='--',alpha = 0.5)

axs[3][2].scatter(data=hist_flow[(hist_flow['model']=='LSTM') & (hist_flow['tag']=='4-6')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[3])
axs[3][2].axline(xy1=(0,0), slope=1, color='#555555')
axs[3][2].set_xlabel('Truth')
axs[3][2].set_xlim(-1.2,25)
axs[3][2].set_ylim(-1.2,25)
# axs[3][2].set_aspect('equal')
axs[3][2].grid(True, linestyle='--',alpha = 0.5)

axs[3][3].scatter(data=hist_flow[(hist_flow['model']=='LSTM') & (hist_flow['tag']=='6-8')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[3])
axs[3][3].axline(xy1=(0,0), slope=1, color='#555555')
axs[3][3].set_xlabel('Truth')
axs[3][3].set_xlim(-1.2,25)
axs[3][3].set_ylim(-1.2,25)
# axs[3][3].set_aspect('equal')
axs[3][3].grid(True, linestyle='--',alpha = 0.5)

plt.tight_layout()
plt.show()
fig.savefig(f'output/{id}/historical_everything_qqplot.png', dpi=300)





##--Future--##
fig,axs=plt.subplots(4,4, figsize=(9,8))
#precip
axs[0][0].scatter(data=future_precip[future_precip['tag']=='0-2'],
                  x='truth',y='PRECIP',alpha=0.2,color=palette[0])
axs[0][0].axline(xy1=(0,0), slope=1, color='#555555')
axs[0][0].set_title('MAP RMSE: 0-2')
axs[0][0].set_ylabel('Interpolated MAP')
axs[0][0].set_xlim(-10,250)
axs[0][0].set_ylim(-10,250)
axs[0][0].grid(True, linestyle='--',alpha = 0.5)

axs[0][1].scatter(data=future_precip[future_precip['tag']=='2-4'],
                  x='truth',y='PRECIP',alpha=0.2,color=palette[0])
axs[0][1].axline(xy1=(0,0), slope=1, color='#555555')
axs[0][1].set_title('MAP RMSE: 2-4')
axs[0][1].set_xlim(-10,250)
axs[0][1].set_ylim(-10,250)
axs[0][1].grid(True, linestyle='--',alpha = 0.5)

axs[0][2].scatter(data=future_precip[future_precip['tag']=='4-6'],
                  x='truth',y='PRECIP',alpha=0.2,color=palette[0])
axs[0][2].axline(xy1=(0,0), slope=1, color='#555555')
axs[0][2].set_title('MAP RMSE: 4-6')
axs[0][2].set_xlim(-10,250)
axs[0][2].set_ylim(-10,250)
axs[0][2].grid(True, linestyle='--',alpha = 0.5)


axs[0][3].scatter(data=future_precip[future_precip['tag']=='6-8'],
                  x='truth',y='PRECIP',alpha=0.2,color=palette[0])
axs[0][3].axline(xy1=(0,0), slope=1, color='#555555')
axs[0][3].set_title('MAP RMSE: 6-8')
axs[0][3].set_aspect('equal')
axs[0][3].set_xlim(-10,250)
axs[0][3].set_ylim(-10,250)
axs[0][3].grid(True, linestyle='--',alpha = 0.5)


#hbv recalibrated
axs[1][0].scatter(data=future_flow[(future_flow['model']=='HBV Re') & (future_flow['tag']=='0-2')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[1])
axs[1][0].axline(xy1=(0,0), slope=1, color='#555555')
axs[1][0].set_ylabel('Simulated HBV-Re flow')
axs[1][0].set_xlim(-1.2,50)
axs[1][0].set_ylim(-1.2,50)
axs[1][0].grid(True, linestyle='--',alpha = 0.5)


axs[1][1].scatter(data=future_flow[(future_flow['model']=='HBV Re') & (future_flow['tag']=='2-4')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[1])
axs[1][1].axline(xy1=(0,0), slope=1, color='#555555')
axs[1][1].set_xlim(-1.2,50)
axs[1][1].set_ylim(-1.2,50)
axs[1][1].grid(True, linestyle='--',alpha = 0.5)

axs[1][2].scatter(data=future_flow[(future_flow['model']=='HBV Re') & (future_flow['tag']=='4-6')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[1])
axs[1][2].axline(xy1=(0,0), slope=1, color='#555555')
axs[1][2].set_xlim(-1.2,50)
axs[1][2].set_ylim(-1.2,50)
axs[1][2].grid(True, linestyle='--',alpha = 0.5)

axs[1][3].scatter(data=future_flow[(future_flow['model']=='HBV Re') & (future_flow['tag']=='6-8')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[1])
axs[1][3].axline(xy1=(0,0), slope=1, color='#555555')
axs[1][3].set_xlim(-1.2,50)
axs[1][3].set_ylim(-1.2,50)
axs[1][3].grid(True, linestyle='--',alpha = 0.5)

#HYMOD
axs[2][0].scatter(data=future_flow[(future_flow['model']=='HYMOD') & (future_flow['tag']=='0-2')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[2])
axs[2][0].axline(xy1=(0,0), slope=1, color='#555555')
axs[2][0].set_ylabel('Simulated HYMOD flow')
axs[2][0].set_xlim(-1.2,50)
axs[2][0].set_ylim(-1.2,50)
axs[2][0].grid(True, linestyle='--',alpha = 0.5)

axs[2][1].scatter(data=future_flow[(future_flow['model']=='HYMOD') & (future_flow['tag']=='2-4')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[2])
axs[2][1].axline(xy1=(0,0), slope=1, color='#555555')
axs[2][1].set_xlim(-1.2,50)
axs[2][1].set_ylim(-1.2,50)
axs[2][1].grid(True, linestyle='--',alpha = 0.5)

axs[2][2].scatter(data=future_flow[(future_flow['model']=='HYMOD') & (future_flow['tag']=='4-6')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[2])
axs[2][2].axline(xy1=(0,0), slope=1, color='#555555')
axs[2][2].set_xlim(-1.2,50)
axs[2][2].set_ylim(-1.2,50)
axs[2][2].grid(True, linestyle='--',alpha = 0.5)

axs[2][3].scatter(data=future_flow[(future_flow['model']=='HYMOD') & (future_flow['tag']=='6-8')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[2])
axs[2][3].axline(xy1=(0,0), slope=1, color='#555555')
axs[2][3].set_xlim(-1.2,50)
axs[2][3].set_ylim(-1.2,50)
axs[2][3].grid(True, linestyle='--',alpha = 0.5)

#LSTM
axs[3][0].scatter(data=future_flow[(future_flow['model']=='LSTM') & (future_flow['tag']=='0-2')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[3])
axs[3][0].axline(xy1=(0,0), slope=1, color='#555555')
axs[3][0].set_ylabel('Simulated LSTM flow')
axs[3][0].set_xlabel('Truth')
axs[3][0].set_xlim(-1.2,50)
axs[3][0].set_ylim(-1.2,50)
axs[3][0].grid(True, linestyle='--',alpha = 0.5)

axs[3][1].scatter(data=future_flow[(future_flow['model']=='LSTM') & (future_flow['tag']=='2-4')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[3])
axs[3][1].axline(xy1=(0,0), slope=1, color='#555555')
axs[3][1].set_xlabel('Truth')
axs[3][1].set_xlim(-1.2,50)
axs[3][1].set_ylim(-1.2,50)
axs[3][1].grid(True, linestyle='--',alpha = 0.5)

axs[3][2].scatter(data=future_flow[(future_flow['model']=='LSTM') & (future_flow['tag']=='4-6')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[3])
axs[3][2].axline(xy1=(0,0), slope=1, color='#555555')
axs[3][2].set_xlabel('Truth')
axs[3][2].set_xlim(-1.2,50)
axs[3][2].set_ylim(-1.2,50)
axs[3][2].grid(True, linestyle='--',alpha = 0.5)

axs[3][3].scatter(data=future_flow[(future_flow['model']=='LSTM') & (future_flow['tag']=='6-8')],
                  x='truth',y='streamflow',alpha=0.2,color=palette[3])
axs[3][3].axline(xy1=(0,0), slope=1, color='#555555')
axs[3][3].set_xlabel('Truth')
axs[3][3].set_xlim(-1.2,50)
axs[3][3].set_ylim(-1.2,50)
axs[3][3].grid(True, linestyle='--',alpha = 0.5)

plt.tight_layout()
plt.show()
fig.savefig(f'output/{id}/future_everything_qqplot.png', dpi=300)