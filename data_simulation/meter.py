import pandas as pd
import numpy as np
import glob
import random
import math
from datetime import datetime
import time
import plotly.graph_objects as go
import plotly
import os
import matplotlib.pyplot as plt
from utils import assign_period, df_format

def adjust_volume(og, transaction_df, leak_rate, start_date, stop_date, errored_pump, reversedSC, tck):
    def interpolate_height(volume):
        for i, model in enumerate(reversedSC):
            if volumes[i] <= volume <= volumes[i + 1]:
                return np.polyval(model, volume)

    og['ClosingStock_readjusted'] = og['ClosingStock_Ini']
    og['OpeningStock_readjusted'] = og['OpeningStock_Ini']
    og['OpeningHeight_readjusted'] = og['OpeningHeight_Ini']
    og['ClosingHeight_readjusted'] = og['ClosingHeight_Ini']
    og['ClosingStock_tc_readjusted'] = og['ClosingStock_Ini_tc']
    og['OpeningStock_tc_readjusted'] = og['OpeningStock_Ini_tc']
    og['ClosingHeight_tc_readjusted'] = og['ClosingStock_tc_readjusted'].apply(interpolate_height)
    og['OpeningHeight_tc_readjusted'] = og['ClosingHeight_tc_readjusted'].shift(fill_value=0)
    og['Var_tc_readjusted'] = og['Var_tc_red']
    og['Var_readjusted'] = og['Var_red']

    before = og[(og['Time_DN'] < start_date)].copy()
    leaking = og[(og['Time_DN'] >= start_date) & (og['Time_DN'] <= stop_date)].copy()
    transaction_df['pump_start_DN'] = [datetime.strptime(dt, '%Y-%m-%d %H:%M:%S').timestamp() for dt in transaction_df['pump_start_dt']]
    transaction_df['pump_stop_DN'] = [datetime.strptime(dt, '%Y-%m-%d %H:%M:%S').timestamp() for dt in transaction_df['pump_stop_dt']]
    errored_transaction = transaction_df[(transaction_df['pump_start_DN'] >= start_date) & (transaction_df['pump_stop_DN'] <= stop_date) & (transaction_df['sticker_number'] == errored_pump)]
    errored_transaction = errored_transaction.sort_values(by='pump_stop_dt')
    # update closing stock and starting stock during the leaking period
    cum_var, cum_var_tc = 0, 0
    for idx, row in errored_transaction.iterrows():
        leakage = -leak_rate*row['volume']*0.01
        leakage_tc = leakage * (1 + tck * (15 - leaking.at[idx, 'TankTemp']))
        cum_var += leakage
        cum_var_tc += leakage_tc

        #find the corresponding 30mins window
        stop_TS = row.pump_stop_DN
        try:
            matching_row = leaking[(leaking['Time_DN'].shift(1) <= stop_TS) & (leaking['Time_DN'] > stop_TS)].iloc[-1]
        except:
            matching_row = leaking.iloc[0]
        curr = matching_row.name
        leaking.at[curr, 'Var_readjusted'] += leakage
        leaking.at[curr, 'Var_tc_readjusted'] += leakage_tc
        leaking.at[curr, 'ClosingStock_readjusted'] = matching_row.ClosingStock_readjusted + cum_var
        leaking.at[curr, 'ClosingHeight_readjusted'] = interpolate_height(matching_row.ClosingStock_readjusted)
        leaking.at[curr, 'ClosingStock_tc_readjusted'] = matching_row.ClosingStock_tc_readjusted + cum_var_tc
        leaking.at[curr, 'ClosingHeight_tc_readjusted'] = interpolate_height(matching_row.ClosingStock_tc_readjusted)
        if idx > leaking.index[0]:
            leaking.at[curr, 'OpeningStock_readjusted'] = leaking.loc[curr - 1, 'ClosingStock_readjusted']
            leaking.at[curr, 'OpeningHeight_readjusted'] = leaking.loc[curr - 1, 'ClosingHeight_readjusted']
            leaking.at[curr, 'OpeningStock_tc_readjusted'] = leaking.loc[curr - 1, 'ClosingStock_tc_readjusted']
            leaking.at[curr, 'OpeningHeight_tc_readjusted'] = leaking.loc[curr - 1, 'ClosingHeight_tc_readjusted']

    after = og[og['Time_DN'] > stop_date].copy()
    res = pd.concat([before, leaking, after])
    return res

AVG_me_rate = [0.5, 1.0, 1.5, 2.0, 2.5]
MONTH = 2629743
C = 0
folder = 'C:/Users/Administrator/OneDrive - RMIT University/RQ3 dataset/Full/Normal_ver2/*.csv'

if os.path.isfile('./simulate_info.csv'):
    simulate_info = pd.read_csv('simulate_info.csv', index_col=0).reset_index(drop=True)
    C = simulate_info.index[-1] + 1
    if simulate_info['Hole_range'].iloc[-1] == 'bottom':
        COUNTER = 1
    elif simulate_info['Hole_range'].iloc[-1] == 'middle':
        COUNTER = 2
    else:
        COUNTER = 0
else:
    simulate_info = pd.DataFrame(columns=['Site', 'Tank', 'Pump', 'ME_rate', 'Start_date', 'Stop_date', 'File_name'])

for i in glob.glob(folder):
    if C == 0:
        C += 1
        continue
    fs_30mins = pd.read_csv(i, index_col=0).reset_index(drop=True)
    Site = i[i.rfind('\\')+1:i.rfind('\\')+5]
    tank = i[i.rfind('Tank')+6]
    grade = i[i.rfind('_')-1]
    tck = 0.000843811 if grade == 4 else 0.00125135

    site_dir = 'F:/Meter error/Pump Cal report/Data/' + Site + '/RQ3/'
    transactions_files = [filename for filename in os.listdir(site_dir) if filename.endswith('.csv') and '_transactions_' in filename]
    dataframes = []
    for filename in transactions_files:
        file_path = os.path.join(site_dir, filename)
        df = pd.read_csv(file_path, index_col=0, skiprows=3).reset_index(drop=True)
        dataframes.append(df)
    transaction_df = pd.concat(dataframes, ignore_index=True)
    inventory_file = [filename for filename in os.listdir(site_dir) if
                      filename.endswith('.csv') and '_inventories_' in filename]
    file_path = os.path.join(site_dir, inventory_file[0])
    inventory_df = pd.read_csv(file_path, index_col=0).reset_index(drop=True)
    inventory_df = inventory_df[(inventory_df['tank_identifier'] == int(tank))]
    inventory_df = inventory_df.drop_duplicates()
    fs_30mins = fs_30mins.drop_duplicates()
    inventory_df = inventory_df.drop_duplicates(subset='event_time', keep='first')
    fs_30mins = fs_30mins.drop_duplicates(subset='Time', keep='first')
    inventory_30ms = fs_30mins.merge(inventory_df[['event_time', 'tc_volume', 'height']], left_on='Time',
                                     right_on='event_time', how='left')
    new_column_names = {
        'tc_volume': 'ClosingStock_Ini_tc',
        'height': 'ClosingHeight_Ini'
    }
    inventory_30ms = inventory_30ms.rename(columns=new_column_names)
    inventory_30ms = inventory_30ms.drop('event_time', axis=1)
    inventory_30ms['OpeningHeight_Ini'] = inventory_30ms['ClosingHeight_Ini'].shift(fill_value=0)
    inventory_30ms['OpeningStock_Ini_tc'] = inventory_30ms['ClosingStock_Ini_tc'].shift(fill_value=0)

    duration = (inventory_30ms['Time_DN'].iloc[-1] - inventory_30ms['Time_DN'].iloc[0]) / MONTH
    transaction_tank = transaction_df[(transaction_df['tank_number'] == int(tank))]
    filtered_df = inventory_30ms.drop_duplicates(subset=['ClosingHeight', 'ClosingStock'])
    filtered_df = filtered_df.sort_values(by='ClosingHeight')

    heights = np.array(filtered_df['ClosingHeight'])
    volumes = np.array(filtered_df['ClosingStock'])

    # Create a list to hold linear models for each segment
    reversedSC = []

    # Iterate through the segments and fit linear models
    for i in range(len(heights) - 1):
        y_segment = np.array([heights[i], heights[i + 1]])
        x_segment = np.array([volumes[i], volumes[i + 1]])

        # Fit a linear model (1st-degree polynomial)
        coefficients = np.polyfit(x_segment, y_segment, 1)
        reversedSC.append(coefficients)

    # for rate in AVG_me_rate:
    rate = 2.0
    if (duration >= 24):
        start_date = inventory_30ms['Time_DN'].iloc[0] + np.random.uniform(15*MONTH, 18*MONTH)
        stop_date = np.random.uniform(start_date+3*MONTH, start_date+6*MONTH)
    elif (duration > 18) and (duration < 24):
        start_date = inventory_30ms['Time_DN'].iloc[0] + np.random.uniform(12*MONTH, 15*MONTH)
        stop_date = np.random.uniform(start_date+2*MONTH, start_date+3*MONTH)
    elif (duration > 12) and (duration < 18):
        start_date = inventory_30ms['Time_DN'].iloc[0] + np.random.uniform(8*MONTH, 10*MONTH)
        stop_date = np.random.uniform(start_date+1*MONTH, inventory_30ms['Time_DN'].iloc[-1] - 1 * MONTH)
    else:
        break
    start_date, stop_date = int(start_date), int(stop_date)
    errored_pump = random.choice(transaction_tank.sticker_number.unique())
    induced_df = adjust_volume(inventory_30ms, transaction_tank, rate, start_date, stop_date, errored_pump, reversedSC, tck)
    induced_df = assign_period(induced_df)
    # induced_df['var/sales'] = induced_df['Var_tc_readjusted'] / induced_df['Sales_Ini_tc'].replace(0, 1)
    final = df_format(induced_df)

    file_name = Site + '_' + tank + "_" + str(rate) + '_' + str(errored_pump) + '_ME'
    final.to_csv(file_name + '.csv')
    # idle = 0, transaction = 1, delivery = 2
    color_map = {
        0: 'red',
        1: 'blue',
        2: 'green'
    }

    scatter = []
    for category, color in color_map.items():
        category_df = final[final['period'] == category]
        scatter.append(
            go.Scatter(
                x=final['Time'],
                # y=category_df['Var_tc_readjusted'],
                y=final['var/sales'],
                mode='markers',
                name=category,
                marker=dict(
                    size=10,
                    color=color,
                    opacity=0.8,
                    showscale=False
                )
            )
        )

    # Define the layout of the plot
    layout = go.Layout(
        title=str(datetime.fromtimestamp(start_date)) + '_' + str(datetime.fromtimestamp(stop_date)),
        xaxis=dict(title='Time'),
        yaxis=dict(title='VAR_adjusted'),
        showlegend=True
    )

    # Create the figure and show the plot
    fig = go.Figure(data=scatter, layout=layout)
    # fig.show()
    plotly.offline.plot(fig, filename=file_name + '.html', auto_open=False)
    simulate_info.loc[C] = [Site, tank, errored_pump, rate, start_date, stop_date, file_name]
    C += 1

simulate_info.to_csv('bottom005_info.csv')


