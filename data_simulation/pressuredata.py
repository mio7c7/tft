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

folder = 'C:/Users/Administrator/OneDrive - RMIT University/RQ3 dataset/pressure/*.csv'
dataframes = []
for i in glob.glob(folder):
    pre = pd.read_csv(i, index_col=0, skiprows=3).reset_index(drop=True)
    dataframes.append(pre)

pre_df = pd.concat(dataframes, ignore_index=True)
idle = pre_df[(pre_df['Enabled'] == True) & (pre_df['Pump On'] == False)]
transaction = pre_df[pre_df['Test Status'] == 'Dispensing']
avg_idle_pressure = idle['Reading'].mean()
avg_transaction_pressure = transaction['Reading'].mean()
print(avg_idle_pressure, avg_transaction_pressure)