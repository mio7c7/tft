import sys
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
# import TSCP2 as cp2
# import losses as ls
import lightning.pytorch as pl
from utils.DataLoader import load_data
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
# from utils.estimate_CPD import estimate_CPs
import torch

path = 'C:/Users/Administrator/Documents/GitHub/tft/pytorch_forecasting/CPD/tl_test/trial_0/epoch=46.ckpt'
best_tft = TemporalFusionTransformer.load_from_checkpoint(path)
batch_size = 128
max_prediction_length = 2*24 #the goal is to make a one-day forecast 48
max_encoder_length = 3*2*24
window_size = 10
tlgrouths = pd.read_csv('C:/Users/Administrator/Documents/GitHub/tft/data_simulation/tankleakage_info.csv',
                        index_col=0).reset_index(drop=True)
test_sequence = pd.read_csv('tankleak.csv')
test_sequence = test_sequence.drop(columns=["Month", "Year", "Season"])
test_sequence['period'] = test_sequence['period'].astype(str)
for tank_sample_id in list(test_sequence['group_id'].unique()):
    if tank_sample_id != "A205_2":
        continue
    sample = test_sequence[(test_sequence['group_id'] == tank_sample_id)]
    df = sample[sample['period'] == '0']
    column_name = 'Var_tc_readjusted'
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    df['Var_tc_readjusted'] = df['Var_tc_readjusted'].rolling(window=window_size, min_periods=1).mean()
    df = df.reset_index(drop=True)
    df['time_idx'] = df.index
    max = df["time_idx"].max() - max_prediction_length
    test_data = TimeSeriesDataSet(
        df[lambda x: (x.time_idx < max)],
        time_idx="time_idx",
        target="Var_tc_readjusted",  # variance
        group_ids=["group_id"],  # tank id
        min_encoder_length=max_encoder_length,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["group_id"],  # tank id, tank location state
        static_reals=["tank_max_height", "tank_max_volume"],
        # tank max height, tank max volume, no. of pumps attached to the tank
        time_varying_known_categoricals=["Time_of_day"],  # season, month, "Month", "Year", "Season"
        time_varying_known_reals=["time_idx"],  # time_idx,
        time_varying_unknown_categoricals=["period"],  # period (idle, transaction, delivery)
        time_varying_unknown_reals=[
            "Var_tc_readjusted",
            "ClosingHeight_tc_readjusted",
            "ClosingStock_tc_readjusted",
            "TankTemp",
        ],  # variance, volume, height, sales(-), delivery(+), temperature, "Del_tc", "Sales_Ini_tc",
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    test = TimeSeriesDataSet.from_dataset(test_data,  df, stop_randomization=True)
    test_dataloader = test.to_dataloader(train=False, batch_size=128, num_workers=0)
    predictions = best_tft.predict(test_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
    pred = predictions.output["prediction"]
    onepred = predictions.output["prediction"][:,:,3]
    onetarget = predictions.x["decoder_target"][:,:]
    mase_values = np.zeros(onepred.shape[0])
    for i in range(onepred.shape[0]):
        abs_diff = torch.abs(onepred[i] - onetarget[i])
        mae = torch.mean(abs_diff)
        denominator = torch.mean(torch.abs(onetarget[i] - np.roll(onetarget[i], 1)))
        scaled_error = mae / denominator
        mase_values[i] = scaled_error
    mse_values = torch.mean((onepred - onetarget)**2, dim=1)
    padded_tensor = np.pad(mse_values, (max_encoder_length, max_prediction_length-1), 'constant')
    xs = pd.to_datetime(df['Time'])
    padded_pred = np.pad(predictions.output["prediction"][:, 1, 3], (max_encoder_length, max_prediction_length - 1),
                         'constant')

    site_id = tank_sample_id[:4]
    tank_id = tank_sample_id[-1]
    tank_info = tlgrouths[(tlgrouths['Site'] == site_id) & (tlgrouths['Tank'] == int(tank_id))]
    startdate = tank_info.iloc[0]['Start_date']
    stopdate = tank_info.iloc[0]['Stop_date']
    temp_df = df[df['Time_DN'] > startdate]
    startindex = temp_df.iloc[0]['time_idx']
    temp_df = df[df['Time_DN'] > stopdate]
    stopindex = temp_df.iloc[0]['time_idx']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    ax1.plot(xs, padded_tensor, label="mse")
    ax1.axvline(x=pd.to_datetime(df.loc[startindex,'Time']), color='r', linestyle='--', label='start')
    ax1.axvline(x=pd.to_datetime(df.loc[stopindex,'Time']), color='r', linestyle='--', label='stop')
    ax1.legend()

    ax2.scatter(xs, df['Var_tc_readjusted'])
    ax2.scatter(xs, padded_pred)
    ax2.set_title('input')
    plt.tight_layout()
    # plt.show()
    plt.savefig(tank_sample_id + '.png')



# y_hat = []
# for i in range(pred.shape[0]-max_prediction_length+1):
#     y_hat.append(pred.data[i, 0, 3].numpy().min())
# fig, ax = plt.subplots()
# ax.plot(xs, actual, label="actual")
# ax.plot(xs, y_hat, label="prediction")
# ax.legend()
# plt.show()
# plotter(xs, y_hat, label="predicted", c=pred_color)
# plotter(xs, actual, label="predicted", c=pred_color)

