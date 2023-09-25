import pandas as pd
import numpy as np
import glob
import math
import datetime
import time
from per_label import assign_period, generate_plots
import plotly.graph_objects as go
import plotly
import os
# os.environ["R_HOME"] = r"C:\\Program Files\\R\\R-4.1.0"
# os.environ["PATH"] = r"C:\\Program Files\\R\\R-4.1.0\\bin\\x64" + ";" + os.environ["R_HOME"]
os.environ["R_HOME"] = 'C:/Program Files/R/R-4.1.0'  # Your R version here 'R-4.0.3'
os.environ["PATH"] = "C:/Program Files/R/R-4.1.0/bin/x64" + ";" + os.environ["PATH"]
import rpy2.robjects as robjects
# Initialisation, settings
GAL = 3.78541
AVG_leak_rate = {'0.05GAL': (0.7*(GAL*.05), 1.3*(GAL*.05))}
hole_range = {'bottom': (0, 0), 'middle': (0.2, 0.5), 'top': (0.5, 0.75)}
folder = 'D:/calibrated_30min/WTAF_WSC_csv/normal/*.csv'
MONTH = 2629743

INFO_DF = pd.read_csv('D:/calibrated_30min/WTAF_WSC_csv/info.csv')
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
    simulate_info = pd.DataFrame(columns=['Site', 'Tank', 'Leak_rate', 'Hole_range', 'Hole_height', 'Start_date', 'Stop_date', 'File_name'])
    C = 0
    COUNTER = 0

def adjust_volume(og, hole_height, leak_rate, start_date, stop_date, reversedSC, tck):
    og['ClosingStock_tc_readjusted'] = og['ClosingStock_Caltc']
    og['OpeningStock_tc_readjusted'] = og['OpeningStock_Caltc']
    og['OpeningHeight_readjusted'] = og['OpeningHeight']
    og['ClosingHeight_readjusted'] = og['ClosingHeight']
    og['Var_tc_readjusted'] = og['Var_tc']
    before = og[(og['Time_DN'] < start_date)].copy()

    leaking = og[(og['Time_DN'] >= start_date) & (og['Time_DN'] <= stop_date)].copy()
    # determine hmax, monthly update
    leaking['HMAX'] = 0
    monthlyflag, idx = 0, leaking.index[0]
    while idx <= leaking.index[-1]:
        monthlyflag = leaking.loc[idx, 'Time_DN']
        subset = leaking[(leaking['Time_DN'] >= monthlyflag) & (leaking['Time_DN'] <= monthlyflag + MONTH)]
        max_fil_height = max(subset['ClosingHeight'])
        leaking.loc[idx:subset.index[-1]+1, 'HMAX'] = max_fil_height
        idx = subset.index[-1]+1

    # update closing stock and starting stock during the leaking period
    cum_var = 0
    for idx, row in leaking.iterrows():
        if row.OpeningHeight_readjusted >= hole_height:
            leakage = - (1+tck*(15-row.TankTemp))* 0.5 * leak_rate * math.sqrt((row.OpeningHeight_readjusted - hole_height) / (row.HMAX - hole_height))
            cum_var += leakage
            leaking.at[idx, 'Var_tc_readjusted'] += leakage

        leaking.at[idx, 'ClosingStock_tc_readjusted'] = row.ClosingStock_Caltc + cum_var
        temp = reversedSC['ChangePoint'].loc[lambda x: x >= row.ClosingStock_Caltc + cum_var]
        if len(temp.index) != 0:
            cp = temp.index[0]
        else:
            cp = reversedSC.index[-1]
        leaking.at[idx, 'ClosingHeight_readjusted'] = reversedSC.loc[cp, 'Intercept'] + reversedSC.loc[cp, 'B1']*leaking.loc[idx, 'ClosingStock_tc_readjusted']

        if idx == leaking.index[0]:
            leaking.at[idx, 'OpeningStock_tc_readjusted'] = og.loc[idx-1, 'ClosingStock_tc_readjusted']
            leaking.at[idx, 'OpeningHeight_readjusted'] = og.loc[idx-1, 'ClosingHeight_readjusted']
        else:
            leaking.at[idx, 'OpeningStock_tc_readjusted'] = leaking.loc[idx-1, 'ClosingStock_tc_readjusted']
            leaking.at[idx, 'OpeningHeight_readjusted'] = leaking.loc[idx-1, 'ClosingHeight_readjusted']

    after = og[og['Time_DN'] > stop_date].copy()
    res = pd.concat([before, leaking, after])
    return res

def preprocess(df):
    transactions, idles, deliveries = assign_period(df)
    transactions['Cumsum_vartc_readjusted'] = transactions['Var_tc_readjusted'].cumsum()
    idles['Cumsum_vartc_readjusted'] = idles['Var_tc_readjusted'].cumsum()
    deliveries['Cumsum_vartc_readjusted'] = deliveries['Var_tc_readjusted'].cumsum()
    return transactions, idles, deliveries

def add(fig, transactions, idles, deliveries):
    fig.add_trace(go.Scatter(x=transactions['Time'], y=transactions['Cumsum_vartc_readjusted'],
                             mode='markers',
                             fillcolor='green',
                             opacity=0.7,
                             name='transactions'))
    fig.add_trace(go.Scatter(x=idles['Time'], y=idles['Cumsum_vartc_readjusted'],
                             mode='markers',
                             fillcolor='blue',
                             opacity=0.7,
                             name='idles'))
    fig.add_trace(go.Scatter(x=deliveries['Time'], y=deliveries['Cumsum_vartc_readjusted'],
                             mode='markers',
                             fillcolor='red',
                             opacity=0.7,
                             name='deliveries'))
    return fig

for i in glob.glob(folder):
    df = pd.read_csv(i, index_col=0).reset_index(drop=True)
    Site = i[i.rfind('\\')+1:i.rfind('\\')+5]
    tank = i[i.rfind('\\')+1:]
    tank = tank[:tank.find('_')]
    tank_no = int(tank[12])
    grade = int(tank[-1])
    tck = 0.000843811 if grade == 4 else 0.00125135

    k = 'F:/Meter error/Pump Cal report/Data/'+ Site + '/' + Site + '_ACal.RDATA'
    info = INFO_DF[(INFO_DF['Site'] == Site) & (INFO_DF['Tank'] == tank_no)]
    GET_PARTIAL = info['Partial'].iloc[0]

    # generate strapping chart (Volume -> height)
    robjects.r['load'](k)
    matrix = robjects.r['Cal_Output']
    names = matrix.names
    for name in names:
        if name[:4] == Site and int(name[12]) == tank_no:
            nt = name
    ob = matrix.rx2(nt).rx2('ND_AMB').rx2('Strap').rx2('Coeff_MinErr')
    array = np.array(ob)
    sc = pd.DataFrame(data=array,
                      columns=['Intercept', 'B1', 'ChangePoint', 'Count'])
    reversedSC = sc.copy()
    for idx, row in sc.iterrows():
        reversedSC.loc[idx, 'ChangePoint'] = row.Intercept + row.B1*row.ChangePoint
        reversedSC.loc[idx, 'B1'] = 1/row.B1
        reversedSC.loc[idx, 'Intercept'] = -1*row.Intercept / row.B1

    # if only use partial data, slice the df
    if GET_PARTIAL == 'T':
        op = time.mktime(datetime.datetime.strptime(info['Start'].iloc[0], "%d/%m/%Y").timetuple())
        ed = time.mktime(datetime.datetime.strptime(info['End'].iloc[0], "%d/%m/%Y").timetuple())
        df = df[(df['Time_DN'] >= op) & (df['Time_DN'] <= ed)].copy()

    duration = (df['Time_DN'].iloc[-1] - df['Time_DN'].iloc[0])/MONTH
    for _, alr in AVG_leak_rate.items():
        leak_rate = np.random.uniform(alr[0], alr[1])
        hr = 'bottom'
        hole_height = np.random.uniform(hole_range.get(hr)[0], hole_range.get(hr)[1])
        if duration >= 18:
            start_date = df['Time_DN'].iloc[0] + np.random.uniform(6*MONTH, 12*MONTH)
            stop_date = np.random.uniform(start_date+3*MONTH, start_date+6*MONTH)
        elif (duration > 12) and (duration < 18):
            start_date = df['Time_DN'].iloc[0] + np.random.uniform(4*MONTH, 6*MONTH)
            stop_date = np.random.uniform(start_date+2*MONTH, start_date+4*MONTH)
        elif (duration < 12) and (duration > 6):
            start_date = df['Time_DN'].iloc[0] + np.random.uniform(2 * MONTH, 3 * MONTH)
            stop_date = np.random.uniform(start_date + 2 * MONTH, df['Time_DN'].iloc[-1] - 1 * MONTH)
        else:
            break
        start_date, stop_date = int(start_date), int(stop_date)

        induced_df = adjust_volume(df, hole_height*max(df['ClosingHeight']), leak_rate, start_date, stop_date, reversedSC, tck)
        transactions, idles, deliveries = preprocess(induced_df)
        fig = generate_plots(transactions, idles, deliveries)
        fig = add(fig, transactions, idles, deliveries)
        file_name = tank + "_" + str(leak_rate) + '_' + hr + str(hole_height)
        induced_df.to_csv(file_name + '.csv')
        fig.update_layout(
            title=tank
        )
        plotly.offline.plot(fig, filename=file_name + '.html', auto_open=False)
        simulate_info.loc[C] = [Site, tank_no, leak_rate, hr, hole_height, start_date, stop_date, file_name]
        C += 1

    COUNTER += 1
    simulate_info.to_csv('bottom005_info.csv')


