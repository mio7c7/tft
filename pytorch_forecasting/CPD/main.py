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
# max_prediction_length = 2 * 24  # the goal is to make a one-day forecast 48
# max_encoder_length = 7 * 2 * 24
# test_sequence["time_idx"] = test_sequence.index
# training_cutoff = test_sequence["time_idx"].max() - max_prediction_length
max_prediction_length = 2*24 #the goal is to make a one-day forecast 48
max_encoder_length = 7*2*24
group = 0# a week 336
# folder = 'C:/Users/Administrator/Documents/GitHub/tft/data_simulation/*_Tank.csv'
folder = 'C:/Users/s3912230/Documents/GitHub/tft/data_simulation/*_Tank.csv'
dfs = []
for i in glob.glob(folder):
    data = pd.read_csv(i, index_col=0).reset_index(drop=True)
    data = data.iloc[:2000]
    data['group_id'] = str(group)
    group += 1
    data["time_idx"] = data.index
    training_cutoff = data["time_idx"].max() - max_prediction_length
    tank_max_height = data["OpeningHeight_readjusted"].max()
    tank_max_volume = data["ClosingHeight_readjusted"].max()
    data['tank_max_height'] = tank_max_height
    data['tank_max_volume'] = tank_max_volume
    data['Month'] = data['Month'].astype(str)
    data['Year'] = data['Year'].astype(str)
    data['period'] = data['period'].astype(str)
    dfs.append(data)
combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df.dropna(subset=['ClosingHeight_readjusted'])
test_sequence = combined_df.dropna(subset=['ClosingHeight_tc_readjusted'])

data_setup = TimeSeriesDataSet(
    test_sequence[lambda x: x.time_idx <= training_cutoff],
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
    time_varying_known_categoricals=["Time_of_day", "Month", "Year", "Season"],  # season, month,
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
)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=3,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=50,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

path = 'C:/Users/s3912230/Documents/GitHub/tft/pytorch_forecasting/lightning_logs/lightning_logs/version_3/checkpoints/epoch=0-step=50.ckpt'
best_tft = TemporalFusionTransformer.load_from_checkpoint(path)

test = TimeSeriesDataSet.from_dataset(data_setup, test_sequence, predict=True, stop_randomization=True)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

predictions = best_tft.predict(test_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
MAE()(predictions.output, predictions.y)
raw_predictions = best_tft.predict(test_dataloader, mode="raw", return_x=True)
for idx in range(5):  # plot 10 examples
    best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True) # grey line is the attention