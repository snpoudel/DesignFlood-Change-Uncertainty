import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def nse(obs, sim):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) between observed and simulated values.
    Removes rows where obs is NaN.
    """
    mask = ~np.isnan(obs)
    obs = obs[mask]
    sim = sim[mask]
    obs_mean = np.mean(obs)
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - obs_mean) ** 2)
    return 1 - (numerator / denominator)

stations = pd.read_csv('ma29basins.csv', dtype={'basin_id': str})
#read each csv file and calculate nse for train period (first:5479) and test period (5479:last)
nse_df = pd.DataFrame(columns=['station_id', 'nse_train', 'nse_test'])
for id in stations['basin_id']:
    df = pd.read_csv(f'output/gr4j_simulated/gr4j_simulated_{id}.csv')
    obs = df['obs'].values
    sim = df['sim'].values
    nse_train = nse(obs[:5479], sim[:5479])
    nse_test = nse(obs[5479:], sim[5479:])
    nse_df = pd.concat([nse_df, pd.DataFrame({'station_id': id, 'nse_train': nse_train, 'nse_test': nse_test}, index=[0])], ignore_index=True)
nse_df.to_csv('output/gr4j_nse.csv', index=False)

#make a cdf plot of train and test nse values

plt.figure(figsize=(6, 4))
for col, color, label in zip(['nse_train', 'nse_test'], ['blue', 'red'], ['Train', 'Test']):
    data = np.sort(nse_df[col])
    cdf = np.arange(1, len(data)+1) / len(data
    )
    plt.plot(data, cdf, color=color, label=label)
    plt.scatter(data, cdf, color=color, s=15, alpha=0.7)  # add points
plt.title('CDF of NSE (Train & Test)')
plt.xlabel('NSE')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.tight_layout()
plt.savefig('output/gr4j_nse_cdf.png')
