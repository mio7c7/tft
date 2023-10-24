import os
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import copy
from pathlib import Path
import warnings
import numpy as np
import glob
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pickle

max_prediction_length = 2*24 #the goal is to make a one-day forecast 48
max_encoder_length = 3*2*24
test_sequence = pd.read_csv('tankleak.csv')
test_sequence = test_sequence.drop(columns=["Month", "Year", "Season"])
test_sequence = test_sequence[test_sequence['period']==0]
test_sequence['period'] = test_sequence['period'].astype(str)
TRAINSIZE = 4000
VALIDSIZE = 500
data = test_sequence[lambda x: x.time_idx <= TRAINSIZE+VALIDSIZE]

processed_dfs = []
column_name = 'Var_tc_readjusted'
groups = test_sequence.groupby('group_id')
window_size = 10
for group_id, group_df in groups:
    Q1 = group_df[column_name].quantile(0.25)
    Q3 = group_df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.75 * IQR
    upper_bound = Q3 + 1.75 * IQR
    group_df = group_df[(group_df[column_name] >= lower_bound) & (group_df[column_name] <= upper_bound)]
    group_df['Var_tc_readjusted'] = group_df['Var_tc_readjusted'].rolling(window=window_size, min_periods=1).mean()
    group_df = group_df.reset_index(drop=True)
    group_df['time_idx'] = group_df.index
    processed_dfs.append(group_df)
final_df = pd.concat(processed_dfs, ignore_index=True)

max_prediction_length = 2*24 #the goal is to make a one-day forecast 48
max_encoder_length = 3*2*24
training = TimeSeriesDataSet(
    final_df[lambda x: x.time_idx <= 4000],
    time_idx="time_idx",
    target="Var_tc_readjusted", #variance
    group_ids=["group_id"], #tank id
    min_encoder_length=max_encoder_length,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group_id"], #tank id, tank location state
    static_reals=["tank_max_height", "tank_max_volume"], #tank max height, tank max volume, no. of pumps attached to the tank
    time_varying_known_categoricals=["Time_of_day"], #season, month, remove "Month", "Year", "Season" if use only a month of data for training
    time_varying_known_reals=["time_idx"], #time_idx,
    time_varying_unknown_categoricals=[],  #  period (idle, transaction, delivery)
    time_varying_unknown_reals=[
        "Var_tc_readjusted",
        "ClosingHeight_tc_readjusted",
        "ClosingStock_tc_readjusted",
        "TankTemp",
    ], # variance, volume, height, sales(-), delivery(+), temperature, "Del_tc", "Sales_Ini_tc",
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
)
validation = TimeSeriesDataSet.from_dataset(training, final_df, predict=True, stop_randomization=True)
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

arly_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="tl_test",
    n_trials=20,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)


# trainer = pl.Trainer(
#     max_epochs=50,
#     accelerator="cpu",
#     enable_model_summary=True,
#     gradient_clip_val=0.1,
#     limit_train_batches=50,  # coment in for training, running valiation every 30 batches
#     # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
#     callbacks=[lr_logger, early_stop_callback],
#     logger=logger,
# )
#
# tft = TemporalFusionTransformer.from_dataset(
#     training,
#     learning_rate=0.03,
#     hidden_size=16,
#     attention_head_size=2,
#     dropout=0.1,
#     hidden_continuous_size=8,
#     loss=QuantileLoss(),
#     log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
#     optimizer="Ranger",
#     reduce_on_plateau_patience=4,
# )
# print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
#
# trainer.fit(
#     tft,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )
#
# path = trainer.checkpoint_callback.best_model_path
# best_tft = TemporalFusionTransformer.load_from_checkpoint(path)
# predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
# MAE()(predictions.output, predictions.y)
# raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
# for idx in range(10):  # plot 10 examples
#     best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)