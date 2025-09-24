import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from SALib.analyze import sobol
from SALib.sample import saltelli

# -------------------
# 1. Data Preprocessing
# -------------------
files = ['scenario3', 'scenario7', 'scenario11', 'scenario15']

merged_df = []
for file in files:
    df = pd.read_csv(f"data/{file}.csv", dtype={'station': str})
    df['error_50'] = df['true_change_50yr_flood'] - df['est_change_50yr_flood']
    df['scenario'] = file
    df = df[['station', 'model', 'precip_rmse', 'error_50', 'scenario']]
    merged_df.append(df.dropna())

merged_df = pd.concat(merged_df, ignore_index=True)

# Encode categorical variables
model_map = {
    0: "HBV Recalib",
    1: "Full-Hymod",
    2: "Hymod",
    3: "LSTM",
    4: "FULL-HYMOD-LSTM",
    5: "HYMOD-LSTM",
}
scenario_map = {
    0: "scenario3",
    1: "scenario7",
    2: "scenario11",
    3: "scenario15",
}

# -------------------
# 2. Define Sobol Problem
# -------------------
lower_bound, upper_bound = merged_df['precip_rmse'].min(), merged_df['precip_rmse'].max()

problem = {
    "num_vars": 3,
    "names": ["model", "precip", "scenario"],
    "bounds": [
        [0, 5],                # model index
        [lower_bound, upper_bound],  # precip error range
        [0, 3],                # scenario index
    ],
}

# -------------------
# 3. Define Model Function
# -------------------
def flood_error_func(x, df_station):
    """
    Given parameters x = [model_idx, precip_rmse, scenario_idx],
    return error_50 for that station.
    """
    model_idx = int(round(x[0]))
    scenario_idx = int(round(x[2]))
    model = model_map[model_idx]
    scenario = scenario_map[scenario_idx]

    # Find closest row in the data
    subset = df_station[
        (df_station["model"] == model)
        & (df_station["scenario"] == scenario)
    ]

    if subset.empty:
        return np.nan  # no match

    # Use nearest precip_rmse
    row = subset.iloc[(subset['precip_rmse'] - x[1]).abs().argsort().iloc[0]]
    return row["error_50"]

# -------------------
# 4. Run Sobol Analysis (per station)
# -------------------

param_values = saltelli.sample(problem, N=32768, calc_second_order=True) #32768 make N power of 2 to avoid convergence warning

stations = merged_df['station'].unique()
results = {}

for station in stations: #stations[:3]:  # limit to 3 for demo
    df_station = merged_df[merged_df["station"] == station]

    # Evaluate model for all samples
    Y = np.array([flood_error_func(x, df_station) for x in param_values])
    Y = Y[~np.isnan(Y)]  # drop missing matches

    Si = sobol.analyze(problem, Y, print_to_console=False)
    results[station] = Si
# Save results
pd.DataFrame(results).to_csv("output/sobol_sensitivity_results.csv", index=False)

# -------------------
# 5. Total Variance for Each Station
# -------------------
var_df = pd.DataFrame({
    "station": list(results.keys()),
    "var": [results[s]["ST"].sum() for s in results.keys()],
})
var_df = var_df.sort_values("var", ascending=False)
station_lat_lon = pd.read_csv('../data/basinID_withLatLon.csv')
station_lat_lon['STAID'] = station_lat_lon['STAID'].astype(str)
var_df = var_df.merge(station_lat_lon, left_on='station', right_on='STAID', how='left')
#save dataframe
var_df.to_csv("output/total_variance_by_station.csv", index=False)

# plot lat and lon and show dot for var
#read shape file of mass
mass_shape = gpd.read_file('../data/gis_mass/cb_2024_us_state_5m.shp')

#plot
fig, ax = plt.subplots(figsize=(6, 4))
mass_shape.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5)

sc = ax.scatter(var_df['LONG_CENT'], var_df['LAT_CENT'], c=var_df['var'],
                 cmap='inferno_r', edgecolors='grey', linewidths=0.1, s=50)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.15, aspect=40)
cbar.set_label("Total variance contribution per basin")

# Set axis limits to match the extent of station locations
buffer = 0.15  # degrees, adjust as needed
lon_min, lon_max = var_df['LONG_CENT'].min() - buffer, var_df['LONG_CENT'].max() + buffer
lat_min, lat_max = var_df['LAT_CENT'].min() - buffer, var_df['LAT_CENT'].max() + buffer
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# Show station ID on top of each dot
for _, row in var_df.iterrows():
    ax.text(row['LONG_CENT'], row['LAT_CENT'] + 0.02, str(row['station']),
            ha='center', va='bottom', fontsize=4, color='black')

plt.savefig("figure/total_variance_by_station.jpg", dpi=300)
plt.savefig("figure/total_variance_by_station.svg")
plt.tight_layout()
plt.show()


# -------------------
# 6. Variance Contribution
# -------------------
plot_df = []
for station, Si in results.items():
    # First-order effects
    for i, name in enumerate(problem["names"]):
        plot_df.append({
            "station": station,
            "parameter": name,
            "value": Si["S1"][i]
        })
    
    # Second-order interactions
    for i in range(len(problem["names"])):
        for j in range(i+1, len(problem["names"])):
            val = Si["S2"][i, j]
            if not np.isnan(val):
                plot_df.append({
                    "station": station,
                    "parameter": f"{problem['names'][i]}+{problem['names'][j]}",
                    "value": val
                })

# Create DataFrame for plotting
plot_df = pd.DataFrame(plot_df)
pivot_df = plot_df.pivot(index="station", columns="parameter", values="value").fillna(0)
pivot_df = pivot_df.clip(lower=0)
#order rows by total absolute variance contribution var from var_df
ordered_rows = var_df.set_index("station")["var"].reindex(pivot_df.index).fillna(0)
pivot_df = pivot_df.loc[ordered_rows.sort_values(ascending=False).index]
# Reorder columns: main effects first, then interactions alphabetically
main_effects = [p for p in problem['names'] if p in pivot_df.columns]
interactions = sorted([c for c in pivot_df.columns if '+' in c])
# ordered_cols = pivot_df.sum(axis=0).sort_values(ascending=False).index.tolist()
# pivot_df = pivot_df[ordered_cols]
# Sort stations by total sensitivity (descending)
# pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]

# ------------------- Plot -------------------
n_colors = len(main_effects) + len(interactions) + 2
palette = sns.color_palette("BrBG", n_colors=n_colors) #BrBG
pivot_df.plot(
    kind='bar',
    stacked=True,
    figsize=(9, 6),
    color=palette,
    edgecolor='grey',
    linewidth=0.1  # Decrease edge size
)
plt.ylabel("Sobol Sensitivity Index (Variance contribution)")
plt.ylim(0, 1.1)
plt.xlabel("Station ID (Ordered in descending total variance contribution)")
plt.legend(loc='upper center', ncol=6, fontsize='small')
plt.tight_layout()
plt.savefig("figure/sobol_sensitivity_analysis.jpg", dpi=300)
plt.savefig("figure/sobol_sensitivity_analysis.svg")
plt.show()
