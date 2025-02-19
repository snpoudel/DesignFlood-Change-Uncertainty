import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV file
df = pd.read_csv("Z:/MA-Precip-Uncertainty-GaugeData/output/allbasins_difference_tyr_flood_modified.csv")

#filter precip zero
df_zeroprecip = df[df['precip_rmse'] == 0] 
df_zeroprecip['precip_category'] = '0'

df = df[df['precip_rmse'] != 0] #filter everything except precip zero
#convert precipitation error into categorical group
df['precip_category']  = pd.cut(df['precip_rmse'], bins=[0,1,2,3,4,6,8],
                        labels=['0-1', '1-2', '2-3', '3-4', '4-6', '6-8'])
#merge back zero precips
df = pd.concat([df,df_zeroprecip], ignore_index=True)
#order the categories
df['precip_category'] = pd.Categorical(df['precip_category'], categories=['0', '0-1', '1-2', '2-3', '3-4', '4-6', '6-8'], ordered=True)

# Create subplots
plt.figure(figsize=(8, 8))
for i, change_col in enumerate(['change_5yr_flood', 'change_10yr_flood', 'change_20yr_flood'], 1):
    plt.subplot(3, 1, i)
    sns.boxplot(
        data=df,
        x='precip_category',
        y=change_col,
        hue='model',
        palette='colorblind',
        showfliers=False,
        legend=False
    )
    plt.xlabel('Precipitation RMSE Category')
    plt.ylabel(f'Change in {change_col.replace("_", " ").title()}')
    plt.xticks(rotation=0)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

#create point plot without lines and more dodging
plt.figure(figsize=(10, 12))
for i, change_col in enumerate(['change_5yr_flood', 'change_10yr_flood', 'change_20yr_flood'], 1):
    plt.subplot(3, 1, i)
    sns.pointplot(
        data=df,
        x='precip_category',
        y=change_col,
        hue='model',
        palette='colorblind',
        errorbar='sd',
        dodge=0.3,  # Increase dodge value for more separation
        markers='o',
        linestyles='--',
        linewidth=1,  # Make the connecting lines thinner
        scale=1.2,  # Increase the size of the points
        errwidth=0.8,  # Increase the width of the error bars
        capsize=0.1,  # Make the error bars smaller
        legend=False # Show legend with brief labels

    )
    plt.xlabel('Precipitation RMSE Category', fontsize=12)
    plt.ylabel(f'Change in {change_col.replace("_", " ").title()}', fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Model')
    plt.title(f'Change in {change_col.replace("_", " ").title()} by Precipitation RMSE Category', fontsize=14)

plt.tight_layout()
plt.show()
