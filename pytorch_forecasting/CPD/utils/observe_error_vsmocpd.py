import numpy as np
import pylab
import statistics
from math import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

if __name__ == '__main__':
    # tlgrouths = pd.read_csv('C:/Users/Administrator/Documents/GitHub/tft/data_simulation/bottom02_info.csv',
    #                         index_col=0).reset_index(drop=True)
    # test_sequence = pd.read_csv('C:/Users/Administrator/Documents/GitHub/tft/pytorch_forecasting/CPD/tl.csv')
    tlgrouths = pd.read_csv('C:/Users/s3912230/Documents/GitHub/tft/pytorch_forecasting/CPD/bottom02_info.csv',
                            index_col=0).reset_index(drop=True)
    test_sequence = pd.read_csv('C:/Users/s3912230/Documents/GitHub/tft/pytorch_forecasting/CPD/tl.csv')
    test_sequence = test_sequence[test_sequence['period'] == 0]
    training_cutoff = 2000 - 96
    mocpd = np.load('mocpd.npy', allow_pickle=True).item()
    mae = np.load('mae_errors31.npy', allow_pickle=True).item()
    rmse = np.load('rmse_errors31.npy', allow_pickle=True).item()
    quantile = np.load('quantile_errors31.npy', allow_pickle=True).item()
    for key, value in mae.items():
        site_id = key[:4]
        tank_id = key[-1]
        tank_info = tlgrouths[(tlgrouths['Site'] == site_id) & (tlgrouths['Tank'] == int(tank_id))]
        tank_sequence = test_sequence[(test_sequence['group_id'] == key)]
        test_seq = tank_sequence.iloc[training_cutoff:]
        test_seq = test_seq.reset_index(drop=True)
        test_seq['time_idx'] = test_seq.index

        startdate = tank_info.iloc[0]['Start_date']
        stopdate = tank_info.iloc[0]['Stop_date']
        temp_df = test_seq[test_seq['Time_DN'] > startdate]
        startindex = temp_df.iloc[0]['time_idx']
        temp_df = test_seq[test_seq['Time_DN'] > stopdate]
        stopindex = temp_df.iloc[0]['time_idx']

        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(18, 36))  # 5 subplots arranged in a single column
        inputs = test_seq['Var_tc_readjusted'].values
        axs[0].plot(inputs, label='TC_var', color='red')
        axs[0].axvline(x=startindex, color='green', linestyle='--')
        axs[0].axvline(x=stopindex, color='green', linestyle='--')
        axs[0].set_title('input')
        axs[0].legend()
    #
        # Plot on the first subplot
        mo = mocpd[key]
        axs[1].plot(mo[0,:], label='errors', color='blue')
        axs[1].plot(mo[1,:], label='threshold', color='orange')
        axs[1].axvline(x=startindex, color='green', linestyle='--')
        axs[1].axvline(x=stopindex, color='green', linestyle='--')
        axs[1].set_title('MOCPD')
        axs[1].legend()

        axs[2].plot(value, label='mae', color='purple')
        axs[2].axvline(x=startindex, color='green', linestyle='--')
        axs[2].axvline(x=stopindex, color='green', linestyle='--')
        axs[2].set_title('mae')
        axs[2].legend()

        rmse_item = rmse[key]
        axs[3].plot(rmse_item, label='rmse', color='green')
        axs[3].axvline(x=startindex, color='green', linestyle='--')
        axs[3].axvline(x=stopindex, color='green', linestyle='--')
        axs[3].set_title('rmse')
        axs[3].legend()

        quantile_item = quantile[key]
        axs[4].plot(quantile_item, label='rmse', color='pink')
        axs[4].axvline(x=startindex, color='green', linestyle='--')
        axs[4].axvline(x=stopindex, color='green', linestyle='--')
        axs[4].set_title('quantile')
        axs[4].legend()


        plt.tight_layout()

        # Show the plot
        plt.savefig(key+'_comparison3.png')