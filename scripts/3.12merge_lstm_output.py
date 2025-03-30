import numpy as np
import pandas as pd
from datetime import date
import os
from concurrent.futures import ThreadPoolExecutor
#read basin list
basin_list = pd.read_csv('data/ma29basins.csv', dtype={'basin_id':str})['basin_id']
coverage = np.append(np.arange(0, 10), [99])
comb = np.arange(0,10)
time_stamp = [date(1, 1, 1), date(50, 1, 1), date(100, 1, 1), date(150, 1, 1), date(200, 1, 1), date(250, 1, 1), date(300, 1, 1), date(350, 1, 1), date(400, 1, 1), date(450, 1, 1), date(500, 1, 1), date(550, 1, 1), date(600, 1, 1), date(650, 1, 1), date(700, 1, 1), date(750, 1, 1), date(800, 1, 1), date(850, 1, 1), date(900, 1, 1), date(950, 1, 1), date(1000, 1, 1)]
time_stamp = [date.isoformat() for date in time_stamp] #convert to isoformat

def process_model_basin_coverage_comb(model, id, cov, com):
    lstm_merged = pd.DataFrame()
    for time in time_stamp:
        # If file exists read lstm output for all time steps and merge them
        if os.path.exists(f'output/{model}/future/lstm_input{id}_coverage{cov}_comb{com}_{time}.csv'):
            lstm_flow = pd.read_csv(f'output/{model}/future/lstm_input{id}_coverage{cov}_comb{com}_{time}.csv')
            lstm_merged = pd.concat([lstm_merged, lstm_flow], ignore_index=True)
            # Save the merged lstm output
            # Keep only one for duplicate entries
            lstm_merged = lstm_merged.drop_duplicates(subset=['date'], keep='first')
    #save the merged lstm output if it exists
    if not lstm_merged.empty:
        lstm_merged.to_csv(f'output/{model}/merged/future/lstm_input{id}_coverage{cov}_comb{com}.csv', index=False)

if __name__ == "__main__":
    model_name = ['regional_lstm', 'regional_lstm_hymod', 'regional_lstm_simp_hymod']
    # model_name = ['regional_lstm']
    with ThreadPoolExecutor() as executor:
        futures = []
        for model in model_name:
            for id in basin_list:
                for cov in coverage:
                    for com in comb:
                        futures.append(executor.submit(process_model_basin_coverage_comb, model, id, cov, com))
        for future in futures:
            future.result()

