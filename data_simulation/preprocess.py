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

# folder = 'C:/Users/Administrator/OneDrive - RMIT University/RQ3 dataset/Full/Normal/*.csv'
folder = 'D:/calibrated_30min/WTAF_WSC_csv/normal/*.csv'

for i in glob.glob(folder):
    fs_30mins = pd.read_csv(i, index_col=0).reset_index(drop=True)
    fs_30mins['Var_red'] = fs_30mins['Var']
    fs_30mins['Var_tc_red'] = fs_30mins['Var_tc']
    flag = False
    cand = []
    for idx, row in fs_30mins.iterrows():
        if not flag:
            if row['Sales_Ini'] != 0 and row['Sales'] == 0.0:
                flag = True
                cand.append((idx, row['Sales_Ini'], row['Sales_Ini_tc']))
            else:
                pass
        else:
            if row['Sales_Ini'] != 0 and row['Sales'] == 0.0:
                cand.append((idx, row['Sales_Ini'], row['Sales_Ini_tc']))
            else:
                cand.append((idx, row['Sales_Ini'], row['Sales_Ini_tc']))
                sum_sales = sum(item[1] for item in cand)
                sum_sales_tc = sum(item[2] for item in cand)
                sum_var = row['Var']
                sum_var_tc = row['Var_tc']
                for t, sales, sales_tc in cand:
                    fs_30mins.at[t, 'Var_red'] = sum_var*sales/sum_sales
                    fs_30mins.at[t, 'Var_tc_red'] = sum_var_tc*sales_tc/sum_sales_tc
                flag = False
                cand = []
    fs_30mins.to_csv(i[i.rfind('\\')+1:])

