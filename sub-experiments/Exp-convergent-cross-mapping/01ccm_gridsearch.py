'''
This script helps with the exploratory analysis to find the optimal parameters (E, Tp, libSize) for convergent cross mapping (CCM).
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyEDM
import itertools

def grid_search_ccm(df, col, target, E_range, Tp_range, libSize_range, sample=100):
    results = []

    for E, Tp, libSize in itertools.product(E_range, Tp_range, libSize_range):
        try:
            ccm_out = pyEDM.CCM(
                dataFrame=df,
                E=E,
                Tp=Tp,
                columns=col,
                target=target,
                libSizes=str(libSize),
                sample=sample
            )
            rho = ccm_out[f"{col}:{target}"].iloc[0]  # take first row since single libSize
            results.append((E, Tp, libSize, rho))
        except Exception as e:
            print(f"Failed for E={E}, Tp={Tp}, libSize={libSize}: {e}")

    res_df = pd.DataFrame(results, columns=["E", "Tp", "libSize", "rho"])
    return res_df

def plot_grid_results(res_df, fixed_lib=None):
    """If fixed_lib is provided, plots only results for that libSize."""
    if fixed_lib is not None:
        res_df = res_df[res_df["libSize"] == fixed_lib]

# --- Main ---
def main():
    # read files
    rehbv8   = pd.read_csv('data/output/rehbv01109060_comb8.csv')[9131:].reset_index(drop=True)
    rehbv_true = pd.read_csv('data/output/hbv_true01109060.csv')[9131:].reset_index(drop=True)

    df = pd.DataFrame({
        'precip': rehbv8['true_precip'],
        'flow(param8)' : rehbv8['qtotal'] - rehbv_true['streamflow']
    })[:5000]

    # Run grid search
    E_range = range(5, 11)
    Tp_range = [0]
    libSize_range = [1000]

    res_df = grid_search_ccm(df, "precip", "flow(param8)", E_range, Tp_range, libSize_range)

    print(res_df.sort_values("rho", ascending=False).head())

    # Plot for one libSize at a time
    for lib in libSize_range:
        plot_grid_results(res_df, fixed_lib=lib)

if __name__ == "__main__":
    main()
