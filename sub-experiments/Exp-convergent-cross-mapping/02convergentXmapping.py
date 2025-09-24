'''
This scripts performs the covergent cross mapping (CCM) analysis
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyEDM
import time
from scipy.stats import pearsonr

def main():
    # read historical files
    rehbv1   = pd.read_csv('data/output/rehbv01109060_comb1.csv')[9131:].reset_index(drop=True)
    rehbv8   = pd.read_csv('data/output/rehbv01109060_comb8.csv')[9131:].reset_index(drop=True)
    rehbv_true = pd.read_csv('data/output/hbv_true01109060.csv')[9131:].reset_index(drop=True)
    # read future files
    # rehbv1   = pd.read_csv('data/output/fu_rehbv01109060_comb1.csv')[9131:].reset_index(drop=True)
    # rehbv8   = pd.read_csv('data/output/fu_rehbv01109060_comb8.csv')[9131:].reset_index(drop=True)
    # rehbv_true = pd.read_csv('data/output/fu_hbv_true01109060.csv')[9131:].reset_index(drop=True)
    # prepare data for convergent cross mapping
    df = pd.DataFrame({
        'precip': rehbv1['true_precip'],
        'flow(param1)' : rehbv1['qtotal'] - rehbv_true['streamflow'],
        'precip': rehbv8['true_precip'],
        'flow(param8)' : rehbv8['qtotal'] - rehbv_true['streamflow']
    })[:1000] #1000

    # CCM: Precip → Output1
    start_time = time.time()
    ccm_out1 = pyEDM.CCM( # ccm_out1 is dataframe with columns ['LibSize', 'precip:flow1', 'flow1:precip']
                          # 'precip:flow1' is cross-map skill (rho) when reconstructing flow1 from precip shadow manifold
                          # cross-map skill (rho) pearson correlation betn CCM predicted flow1 and true flow1
        dataFrame=df, 
        E=10,                #10 Embedding dimension: reconstructs precip's shadow manifold 
                            # E =3 is using vectors of length 3 → [precip_t, precip_{t-1}, precip_{t-2}].
                            # Higher E captures more system dynamics but needs more data.

        Tp=0,               # Prediction horizon (time lag):
                            # Tp=0 → predict flow1 at the same time as the manifold point.
                            # Tp>0 → predict flow1 shifted forward (causal influence into future).
                            # Tp<0 → predict flow1 in the past (testing reverse direction).

        columns="precip",   # Variable used to reconstruct the shadow manifold (potential driver).
        target="flow(param1)",     # Variable to be cross-mapped (potential effect).

        libSizes="20 1000 100", # Range of library sizes (subset of time points used for cross mapping).
                            # Here: test sizes 10, 20, ..., 200.
                            # Larger libSize → more information, better manifold reconstruction.

        sample=100          # For each libSize, take 200 random subsets of that size 
                            # (to reduce sampling bias), then average correlation (ρ).
    )
    print('Time taken for Flow1:', time.time() - start_time)

    # CCM: Precip → Output8
    ccm_out8 = pyEDM.CCM(
        dataFrame=df,
        E=10,
        Tp=0,
        columns="precip",
        target="flow(param8)",
        libSizes="20 1000 100",
        sample=100
    )
    print('Time taken for Flow8:', time.time() - start_time)
    # Pearson correlation between precip and residuals
    corr1, _ = pearsonr(df['precip'], df['flow(param1)'])
    corr8, _ = pearsonr(df['precip'], df['flow(param8)'])
    corr1 = round(corr1, 2)
    corr8 = round(corr8, 2)

    # Two-panel plot: Each panel shows both directions for one param set
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    # Panel 1: Param set 1
    axes[0].plot(ccm_out1['LibSize'], ccm_out1['precip:flow(param1)'], marker='o', markersize=4, label="Precipitation xmap Residual")
    # axes[0].plot(ccm_out1['LibSize'], ccm_out1['flow(param1):precip'], marker='s', markersize=4, label="Residual xmap Precipitation")
    axes[0].set_title("a)Param set 1, NSE:0.99, Δ50yr flood error:-0.72%", fontsize=10)
    axes[0].set_xlabel("Library Size (time-series length)")
    axes[0].set_ylabel("Cross Map Skill (ρ)")
    axes[0].set_ylim(0, 0.5)
    axes[0].legend(fontsize='small', loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    # Place Pearson corr text at top left, just below legend
    # axes[0].text(0.05, 0.8, f"Pearson corr: {corr1}", ha='left', va='top', transform=axes[0].transAxes, fontsize=8)

    # Panel 2: Param set 8
    axes[1].plot(ccm_out8['LibSize'], ccm_out8['precip:flow(param8)'], marker='o', markersize=4, label="Precipitation xmap Residual")
    # axes[1].plot(ccm_out8['LibSize'], ccm_out8['flow(param8):precip'], marker='s', markersize=4, label="Residual xmap Precipitation")
    axes[1].set_title("b)Param set 8, NSE:0.99, Δ50yr flood error:+15.14%", fontsize=10)
    axes[1].set_xlabel("Library Size (time-series length)")
    axes[1].set_ylabel("Cross Map Skill (ρ)")
    axes[1].set_ylim(0, 0.5)
    axes[1].legend(fontsize='small', loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    # axes[1].text(0.05, 0.8, f"Pearson corr: {corr8}", ha='left', va='top', transform=axes[1].transAxes, fontsize=8)

    # fig.suptitle("Convergent Cross Mapping: Precipitation ↔ Flow Residual", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figure/resid_CCM.jpg", dpi=300, bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":   # this helps to execute script in standalone mode
    main()
