import numpy as np
import pylab
import statistics
from math import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

if __name__ == '__main__':
    data_dict = np.load('mae_errors3.npy', allow_pickle=True).item()
    tlgrouths = pd.read_csv('C:/Users/Administrator/Documents/GitHub/tft/data_simulation/bottom02_info.csv',
                            index_col=0).reset_index(drop=True)
    test_sequence = pd.read_csv('C:/Users/Administrator/Documents/GitHub/tft/pytorch_forecasting/CPD/tl.csv')
    # tlgrouths = pd.read_csv('C:/Users/s3912230/Documents/GitHub/tft/pytorch_forecasting/CPD/bottom02_info.csv',
    #                         index_col=0).reset_index(drop=True)
    # test_sequence = pd.read_csv('C:/Users/s3912230/Documents/GitHub/tft/pytorch_forecasting/CPD/tl.csv')
    test_sequence = test_sequence[test_sequence['period'] == 0]
    training_cutoff = 2000 - 96
    for key, value in data_dict.items():
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
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(18, 12))
        inputs = test_seq['Var_tc_readjusted'].values
        ax1.plot(inputs, label='TC_var', color='red')
        ax1.axvline(x=startindex, color='green', linestyle='--')
        ax1.axvline(x=stopindex, color='green', linestyle='--')
        ax1.set_title('input')
        ax1.legend()
    #
        # Plot on the first subplot
        ax2.plot(value, label='errors', color='blue')
        ax2.set_title('Error trend')
        ax2.axvline(x=startindex, color='green', linestyle='--')
        ax2.axvline(x=stopindex, color='green', linestyle='--')
        ax2.legend()

        plt.tight_layout()

        # Show the plot
        plt.savefig(key+'_mar3.png')