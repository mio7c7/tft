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

batch_size = 64
# test_sequence = load_data('C:/Users/s3912230/Documents/GitHub/tft/data_simulation/A082_4_0.757082_Tank.csv')
# test_sequence = test_sequence.dropna(subset=['ClosingHeight_tc_readjusted'])
# test_sequence.reset_index(inplace=True, drop=True)
# test_sequence["time_idx"] = test_sequence.index
# training_cutoff = test_sequence["time_idx"].max() - max_prediction_length
max_prediction_length = 2*24 #the goal is to make a one-day forecast 48
max_encoder_length = 7*2*24
# group = 0# a week 336
# folder = 'C:/Users/Administrator/Documents/GitHub/tft/data_simulation/*_Tank.csv'
# folder = 'C:/Users/s3912230/Documents/GitHub/tft/data_simulation/*_Tank.csv'
# dfs = []
# for i in glob.glob(folder):
#     data = pd.read_csv(i, index_col=0).reset_index(drop=True)
#     file_name = i.split('\\')[-1]
#     parts = file_name.split('_')[:2]
#     group_id = '_'.join(parts)
#     data['group_id'] = group_id
#     data["time_idx"] = data.index
#     training_cutoff = data["time_idx"].max() - max_prediction_length
#     tank_max_height = data["OpeningHeight_readjusted"].max()
#     tank_max_volume = data["ClosingHeight_readjusted"].max()
#     data['tank_max_height'] = tank_max_height
#     data['tank_max_volume'] = tank_max_volume
#     dfs.append(data)
# combined_df = pd.concat(dfs, ignore_index=True)
# test_sequence = combined_df.dropna(subset=['ClosingHeight_readjusted', 'ClosingHeight_tc_readjusted','Var_tc_readjusted','ClosingStock_tc_readjusted'])
# csv_filename = 'tankleak.csv'
# test_sequence.to_csv(csv_filename, index=False)
training_cutoff = 2000 - max_prediction_length
test_sequence = pd.read_csv('tankleak.csv')
test_sequence = test_sequence[(test_sequence['group_id'] == 'A128_5')]
test_sequence = test_sequence.drop(columns=["Month", "Year", "Season"])
test_sequence['period'] = test_sequence['period'].astype(str)
max = test_sequence["time_idx"].max() - max_prediction_length
xs = [i for i in range(max_encoder_length-1, max_encoder_length-1 + 128*12)]
actual = test_sequence[lambda x: (x.time_idx < max_encoder_length-1 + 128*12) & (x.time_idx >= max_encoder_length-1)]['Var_tc_readjusted'].array
test_data = TimeSeriesDataSet(
    test_sequence[lambda x: (x.time_idx <= max)],
    time_idx="time_idx",
    target="Var_tc_readjusted",  # variance
    group_ids=["group_id"],  # tank id
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group_id"],  # tank id, tank location state
    static_reals=["tank_max_height", "tank_max_volume"],
    # tank max height, tank max volume, no. of pumps attached to the tank
    time_varying_known_categoricals=["Time_of_day"],  # season, month, "Month", "Year", "Season"
    time_varying_known_reals=["time_idx"],  # time_idx,
    time_varying_unknown_categoricals=["period"],  # period (idle, transaction, delivery)
    time_varying_unknown_reals=[
        "Var_tc_readjusted",
        "Del_tc",
        "Sales_Ini_tc",
        "ClosingHeight_tc_readjusted",
        "ClosingStock_tc_readjusted",
        "TankTemp",
    ],  # variance, volume, height, sales(-), delivery(+), temperature,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
)

test = TimeSeriesDataSet.from_dataset(test_data,  test_sequence[lambda x: (x.time_idx < 128*12)], stop_randomization=True)
test_dataloader = test.to_dataloader(train=False, batch_size=128, num_workers=0)

# path = 'C:/Users/s3912230/Documents/GitHub/tft/pytorch_forecasting/lightning_logs/lightning_logs/version_0/checkpoints/epoch=0-step=50.ckpt'
path = 'C:/Users/Administrator/Documents/GitHub/tft/pytorch_forecasting/lightning_logs/lightning_logs/version_3/checkpoints/epoch=0-step=50.ckpt'
best_tft = TemporalFusionTransformer.load_from_checkpoint(path)

predictions = best_tft.predict(test_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
pred = predictions.output["prediction"]
y_hat = []
for i in range(pred.shape[0]-max_prediction_length+1):
    y_hat.append(pred.data[i, 0, 3].numpy().min())
fig, ax = plt.subplots()
ax.plot(xs, actual, label="actual")
ax.plot(xs, y_hat, label="prediction")
ax.legend()
plt.show()
# plotter(xs, y_hat, label="predicted", c=pred_color)
# plotter(xs, actual, label="predicted", c=pred_color)

