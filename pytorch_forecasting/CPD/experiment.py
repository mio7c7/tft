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
from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pickle
import matplotlib.pyplot as plt
import math
import argparse
import sys
sys.path.append('./')
from ssa.btgym_ssa import SSA


parser = argparse.ArgumentParser(description='TFT on leakage datra')
parser.add_argument('--max_prediction_length', type=int, default=2 * 24, help='forecast horizon')
parser.add_argument('--max_encoder_length', type=int, default=5 * 2 * 24, help='past reference data')
parser.add_argument('--trainsize', type=int, default=4000, help='train size')
parser.add_argument('--validsize', type=int, default=500, help='validtaion size')
parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')
parser.add_argument('--path', type=str, default='notimeidx_r2_5d2d', help='TensorBoardLogger')
parser.add_argument('--tank_sample_id', type=str, default='A205_1', help='tank sample for experiment')
parser.add_argument('--quantile', type=float, default=0.95, help='threshold quantile')
parser.add_argument('--threshold_scale', type=float, default=1, help='threshold scale')
parser.add_argument('--step', type=int, default=12, help='step')
args = parser.parse_args()

max_prediction_length = args.max_prediction_length
max_encoder_length = args.max_encoder_length
test_sequence = pd.read_csv('pytorch_forecasting/CPD/tl.csv')
# test_sequence = test_sequence.drop(columns=["Month", "Year", "Season"])
test_sequence = test_sequence[test_sequence['period'] == 0]
test_sequence['period'] = test_sequence['period'].astype(str)
TRAINSIZE = args.trainsize
VALIDSIZE = args.validsize
data = test_sequence[lambda x: x.time_idx <= TRAINSIZE + VALIDSIZE]
data = data[abs(data['Var_tc_readjusted']) < args.out_threshold]
# tlgrouths = pd.read_csv('pytorch_forecasting/CPD/tankleakage_info.csv', index_col=0).reset_index(drop=True)

processed_dfs = []
groups = data.groupby('group_id')
window_size = 10
for group_id, group_df in groups:
    group_df = group_df.reset_index(drop=True)
    group_df['time_idx'] = group_df.index
    processed_dfs.append(group_df)
final_df = pd.concat(processed_dfs, ignore_index=True)

training = TimeSeriesDataSet(
    final_df[lambda x: x.time_idx <= 3750],
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
    time_varying_known_categoricals=["Time_of_day"],
    # season, month, remove "Month", "Year", "Season" if use only a month of data for training
    time_varying_known_reals=[],  # time_idx,
    time_varying_unknown_categoricals=[],  # period (idle, transaction, delivery)
    time_varying_unknown_reals=[
        "Var_tc_readjusted",
        "ClosingHeight_tc_readjusted",
        "ClosingStock_tc_readjusted",
        "TankTemp",
    ],  # variance, volume, height, sales(-), delivery(+), temperature, "Del_tc", "Sales_Ini_tc",
    # target_normalizer=GroupNormalizer(
    #     groups=["group_id"], transformation="softplus"
    # ),  # use softplus and normalize by group
    # target_normalizer=EncoderNormalizer(
    #     method='robust',
    #     max_length=None,
    #     center=True,
    #     transformation=None,
    #     method_kwargs={}
    # ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
)
validation = TimeSeriesDataSet.from_dataset(training, final_df, predict=True, stop_randomization=True)
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name=args.path)  # logging results to a tensorboard

study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path=args.path,
    n_trials=20,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(4, 64),
    hidden_continuous_size_range=(4, 64),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.0001, 0.1),
    dropout_range=(0.1, 0.5),
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
#     limit_train_batches=50,  # comment in for training, running valiation every 30 batches
#     # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
#     callbacks=[lr_logger, early_stop_callback],
#     logger=logger,
# )
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
# training_cutoff = 2000 - max_prediction_length
# tank_sequence = test_sequence[(test_sequence['group_id'] == args.tank_sample_id)]
# tank_sequence = tank_sequence[tank_sequence['period'] == '0']
# train_seq = tank_sequence.iloc[:training_cutoff]
# train_seq = train_seq[abs(train_seq['Var_tc_readjusted']) < args.out_threshold]
# train_seq = train_seq.reset_index(drop=True)
# train_seq['time_idx'] = train_seq.index
# X = np.array(train_seq['Var_tc_readjusted'].values)
# ssa = SSA(window=5, max_length=len(X))
# X_pred = ssa.reset(X)
# X_pred = ssa.transform(X_pred, state=ssa.get_state())
# reconstructeds = X_pred.sum(axis=0)
# residuals = X - reconstructeds
# resmean = residuals.mean()
# M2 = ((residuals - resmean) ** 2).sum()
#
# tn = TimeSeriesDataSet.from_dataset(training, train_seq, stop_randomization=True)
# train_dataloader = tn.to_dataloader(train=False, batch_size=128, num_workers=0)
# train_predictions = best_tft.predict(train_dataloader, mode="raw", return_x=True,
#                                      trainer_kwargs=dict(accelerator="cpu"))
# trainpred = train_predictions.output["prediction"][:, :, 3]
# traintarget = train_predictions.x["decoder_target"][:, :]
# MSE = torch.mean((trainpred - traintarget) ** 2, dim=1)
# mse_quantile = np.quantile(MSE, args.quantile)
# final_threshold = args.threshold_scale * mse_quantile
# test_seq = tank_sequence.iloc[training_cutoff:]
# test_seq = test_seq.reset_index(drop=True)
# test_seq['time_idx'] = test_seq.index
# step = args.step
# ctr = 0
# scores = [0] * test_seq.shape[0]
# errors = np.array(MSE)
# thresholds = [final_threshold] * test_seq.shape[0]
# outliers = []
# filtered = []
# while ctr < test_seq.shape[0]:
#     new = test_seq['Var_tc_readjusted'].iloc[ctr:ctr + step].values
#     updates = ssa.update(new)
#     updates = ssa.transform(updates, state=ssa.get_state())[:, 5 - 1:]
#     reconstructed = updates.sum(axis=0)
#     residual = new - reconstructed
#     residuals = np.concatenate([residuals, residual])
#     # start_time = time.time()
#     for i1 in range(len(new)):
#         if new[i1] > 1 or new[i1] < -1:
#             outliers.append(ctr + i1)
#             filtered.append(np.mean(filtered[-5:] if len(filtered) > 5 else 0))
#         else:
#             delta = residual[i1] - resmean
#             resmean += delta / (ctr + i1 + training_cutoff)
#             M2 += delta * (residual[i1] - resmean)
#             stdev = math.sqrt(M2 / (ctr + i1 + training_cutoff - 1))
#             threshold_upper = resmean + 2 * stdev
#             threshold_lower = resmean - 2 * stdev
#
#             if (residual[i1] <= threshold_upper) and (residual[i1] >= threshold_lower):
#                 filtered.append(new[i1])
#             else:
#                 outliers.append(ctr + i1)
#                 filtered.append(np.mean(filtered[-5:] if len(filtered) > 5 else 0))
#     test_seq.loc[ctr:ctr + step - 1, 'Var_tc_readjusted'] = filtered[-step:]
#
#     if ctr >= max_prediction_length + max_encoder_length:
#         new_prediction_data = test_seq[ctr + step - max_prediction_length - max_encoder_length:ctr + step]
#         new_prediction_data = TimeSeriesDataSet.from_dataset(training, new_prediction_data,
#                                                              stop_randomization=True)
#         new_prediction_data = new_prediction_data.to_dataloader(train=False, batch_size=128, num_workers=0)
#         new_raw_predictions = best_tft.predict(new_prediction_data, mode="raw", return_x=True)
#         onepred = new_raw_predictions.output["prediction"][:, :, 3]
#         onetarget = new_raw_predictions.x["decoder_target"][:, :]
#         mse_values = torch.mean((onepred - onetarget) ** 2, dim=1)
#         errors = np.append(errors, mse_values)
#         mse_quantile = np.quantile(errors, args.quantile)
#         final_threshold = args.threshold_scale * mse_quantile
#         thresholds[ctr:ctr + step] = [final_threshold] * step
#         scores[ctr:ctr + step] = [mse_values] * step
#
#     ctr += step
#     if ctr + step >= test_seq.shape[0]:
#         break
#
# site_id = args.tank_sample_id[:4]
# tank_id = args.tank_sample_id[-1]
# tank_info = tlgrouths[(tlgrouths['Site'] == site_id) & (tlgrouths['Tank'] == int(tank_id))]
# startdate = tank_info.iloc[0]['Start_date']
# stopdate = tank_info.iloc[0]['Stop_date']
# temp_df = test_seq[test_seq['Time_DN'] > startdate]
# startindex = temp_df.iloc[0]['time_idx']
# temp_df = test_seq[test_seq['Time_DN'] > stopdate]
# stopindex = temp_df.iloc[0]['time_idx']
#
# filtered = filtered + [0] * (len(scores) - len(filtered))
# ts = pd.to_datetime(test_seq['Time'])
# fig = plt.figure()
# fig, ax = plt.subplots(2, figsize=[18, 16], sharex=True)
#
# ax[0].scatter(ts, filtered)
# ax[0].axvline(x=pd.to_datetime(test_seq.loc[startindex, 'Time']), color='green', linestyle='--')
# ax[0].axvline(x=pd.to_datetime(test_seq.loc[stopindex, 'Time']), color='green', linestyle='--')
# ax[1].scatter(ts, scores)
# ax[1].scatter(ts, thresholds)
# plt.tight_layout()
# plt.savefig(args.tank_sample_id + '.png')


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
