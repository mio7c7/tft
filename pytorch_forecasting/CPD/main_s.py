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
import argparse
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('./')
from ssa.btgym_ssa import SSA
from evaluation import Evaluation_metrics

parser = argparse.ArgumentParser(description='TFT on leakage datra')
parser.add_argument('--max_prediction_length', type=int, default=2 * 24, help='forecast horizon')
parser.add_argument('--max_encoder_length', type=int, default=3 * 2 * 24, help='past reference data')
parser.add_argument('--trainsize', type=int, default=4000, help='train size')
parser.add_argument('--validsize', type=int, default=500, help='validtaion size')
parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')
parser.add_argument('--path', type=str, default='no_norm', help='TensorBoardLogger')
parser.add_argument('--tank_sample_id', type=str, default='A205_1', help='tank sample for experiment')
parser.add_argument('--quantile', type=float, default=0.99, help='threshold quantile')
parser.add_argument('--threshold_scale', type=float, default=1, help='threshold scale')
parser.add_argument('--step', type=int, default=12, help='step')
parser.add_argument('--model_path', type=str,
                    default='/no_normaliser/trial_16/epoch=48.ckpt', help='model_path')
parser.add_argument('--outfile', type=str, default='no_norm', help='step')
args = parser.parse_args()

max_prediction_length = args.max_prediction_length
max_encoder_length = args.max_encoder_length
test_sequence = pd.read_csv('pytorch_forecasting/CPD/tankleak.csv')
test_sequence = test_sequence[test_sequence['period'] == 0]
test_sequence['period'] = test_sequence['period'].astype(str)
TRAINSIZE = args.trainsize
VALIDSIZE = args.validsize
data = test_sequence[lambda x: x.time_idx <= TRAINSIZE + VALIDSIZE]
data = data[abs(data['Var_tc_readjusted']) < args.out_threshold]
tlgrouths = pd.read_csv('pytorch_forecasting/CPD/tankleakage_info.csv',
                        index_col=0).reset_index(drop=True)

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
    time_varying_known_reals=["time_idx"],  # time_idx,
    time_varying_unknown_categoricals=[],  # period (idle, transaction, delivery)
    time_varying_unknown_reals=[
        "Var_tc_readjusted",
        "ClosingHeight_tc_readjusted",
        "ClosingStock_tc_readjusted",
        "TankTemp",
    ],  # variance, volume, height, sales(-), delivery(+), temperature, "Del_tc", "Sales_Ini_tc",
    # target_normalizer=GroupNormalizer(
    #     groups=["group_id"], transformation="relu"
    # ),  # use softplus and normalize by group
    # target_normalizer=GroupNormalizer(
    #         method='standard',
    #         groups=["group_id"],
    #         center=True,
    #         scale_by_group=True,
    #         transformation=None,
    #         method_kwargs={}
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
quantile_levels = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
def loss(y_pred, target):
    # calculate quantile loss
    losses = []
    for i, q in enumerate(quantile_levels):
        errors = target - y_pred[..., i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
    losses = 2 * torch.cat(losses, dim=2)
    return losses

def find_middle_values(values):
    consecutive_sequences = []
    current_sequence = [values[0]]
    for i in range(1, len(values)):
        if values[i] - values[i-1] == 1:
            current_sequence.append(values[i])
        else:
            if len(current_sequence) >= 10:
                middle_index = len(current_sequence) // 2
                middle_value = current_sequence[middle_index]
                consecutive_sequences.append(middle_value)
            current_sequence = [values[i]]
    # Check if there's a sequence at the end
    if len(current_sequence) >= 10:
        middle_index = len(current_sequence) // 2
        middle_value = current_sequence[middle_index]
        consecutive_sequences.append(middle_value)
    return consecutive_sequences

if __name__ == '__main__':
    path = os.getcwd() + args.model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(path)
    training_cutoff = 2000 - max_prediction_length
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []
    runtime = []
    error_margin = 864000
    for tank_sample_id in list(test_sequence['group_id'].unique()):
        # if tank_sample_id != 'F203_5':
        #     continue
        tank_sequence = test_sequence[(test_sequence['group_id'] == tank_sample_id)]
        tank_sequence = tank_sequence[tank_sequence['period'] == '0']
        train_seq = tank_sequence.iloc[:training_cutoff]
        train_seq = train_seq[abs(train_seq['Var_tc_readjusted']) < args.out_threshold]
        train_seq = train_seq.reset_index(drop=True)
        train_seq['time_idx'] = train_seq.index
        X = np.array(train_seq['Var_tc_readjusted'].values)
        ssa = SSA(window=5, max_length=len(X))
        X_pred = ssa.reset(X)
        X_pred = ssa.transform(X_pred, state=ssa.get_state())
        reconstructeds = X_pred.sum(axis=0)
        residuals = X - reconstructeds
        resmean = residuals.mean()
        M2 = ((residuals - resmean) ** 2).sum()

        tn = TimeSeriesDataSet.from_dataset(training, train_seq, stop_randomization=True)
        train_dataloader = tn.to_dataloader(train=False, batch_size=128, num_workers=0)
        train_predictions = best_tft.predict(train_dataloader, mode="raw", return_x=True,
                                             trainer_kwargs=dict(accelerator="gpu"))
        trainpred = train_predictions.output["prediction"][:, :, :]
        traintarget = train_predictions.x["decoder_target"][:, :]
        quantile_loss = loss(trainpred, traintarget)
        quantile_loss = torch.sum(quantile_loss, dim=2)
        quantile_loss, _ = torch.median(quantile_loss, dim=1)
        # MSE = torch.mean((trainpred - traintarget) ** 2, dim=1)
        base = torch.quantile(quantile_loss, args.quantile)
        final_threshold = args.threshold_scale * base
        test_seq = tank_sequence.iloc[training_cutoff:]
        test_seq = test_seq.reset_index(drop=True)
        test_seq['time_idx'] = test_seq.index
        step = args.step
        ts = pd.to_datetime(test_seq['Time'])
        scores = [0] * test_seq.shape[0]
        errors = quantile_loss
        thresholds = [final_threshold] * test_seq.shape[0]
        outliers = []
        filtered = []
        site_id = tank_sample_id[:4]
        tank_id = tank_sample_id[-1]
        tank_info = tlgrouths[(tlgrouths['Site'] == site_id) & (tlgrouths['Tank'] == int(tank_id))]
        startdate = tank_info.iloc[0]['Start_date']
        stopdate = tank_info.iloc[0]['Stop_date']
        temp_df = test_seq[test_seq['Time_DN'] > startdate]
        startindex = temp_df.iloc[0]['time_idx']
        temp_df = test_seq[test_seq['Time_DN'] > stopdate]
        stopindex = temp_df.iloc[0]['time_idx']
        gt_margin = []
        gt_margin.append((ts[startindex-10], ts[startindex] + pd.to_timedelta(7, unit='D'), ts[startindex]))
        gt_margin.append((ts[stopindex-10], ts[stopindex] + pd.to_timedelta(7, unit='D'), ts[stopindex]))
        ctr = 0
        while ctr < test_seq.shape[0]:
            new = test_seq['Var_tc_readjusted'].iloc[ctr:ctr + step].values
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, 5 - 1:]
            reconstructed = updates.sum(axis=0)
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])
            # start_time = time.time()
            for i1 in range(len(new)):
                if new[i1] > 1 or new[i1] < -1:
                    outliers.append(ctr + i1)
                    filtered.append(np.mean(filtered[-5:] if len(filtered) > 5 else 0))
                else:
                    delta = residual[i1] - resmean
                    resmean += delta / (ctr + i1 + training_cutoff)
                    M2 += delta * (residual[i1] - resmean)
                    stdev = math.sqrt(M2 / (ctr + i1 + training_cutoff - 1))
                    threshold_upper = resmean + 2 * stdev
                    threshold_lower = resmean - 2 * stdev

                    if (residual[i1] <= threshold_upper) and (residual[i1] >= threshold_lower):
                        filtered.append(new[i1])
                    else:
                        outliers.append(ctr + i1)
                        filtered.append(np.mean(filtered[-5:] if len(filtered) > 5 else 0))
            test_seq.loc[ctr:ctr + step - 1, 'Var_tc_readjusted'] = filtered[-step:]
            ctr += step
            if ctr + step >= test_seq.shape[0]:
                break

        test = TimeSeriesDataSet.from_dataset(training, test_seq, stop_randomization=True)
        test_dataloader = test.to_dataloader(train=False, batch_size=128, num_workers=0)
        new_raw_predictions = best_tft.predict(test_dataloader, mode="raw", return_x=True,
                                       trainer_kwargs=dict(accelerator="gpu"))
        onepred = new_raw_predictions.output["prediction"][:, :, :]
        onetarget = new_raw_predictions.x["decoder_target"][:, :]
        quantile_loss = loss(onepred, onetarget)
        quantile_loss = torch.sum(quantile_loss, dim=2)
        quantile_loss, _ = torch.median(quantile_loss, dim=1)
        # mse_values = torch.mean((onepred - onetarget) ** 2, dim=1)
        ctr = max_encoder_length
        while ctr < len(quantile_loss)-max_prediction_length:
            mse_ind = ctr-max_encoder_length
            mv = quantile_loss[mse_ind:mse_ind+step]
            errors = torch.cat((errors, mv), dim=0)
            mse_quantile = torch.quantile(errors, args.quantile)
            final_threshold = args.threshold_scale * mse_quantile
            thresholds[ctr:ctr + step] = [final_threshold] * step
            scores[ctr:ctr + step] = mv
            ctr += step
            if ctr + step >= test_seq.shape[0]:
                ss = test_seq.shape[0] - str
                mse_ind = ctr - max_encoder_length
                mv = quantile_loss[mse_ind:mse_ind + ss]
                errors = torch.cat((errors, mv), dim=0)
                mse_quantile = torch.quantile(errors, args.quantile)
                final_threshold = args.threshold_scale * mse_quantile
                thresholds[ctr:ctr + ss] = [final_threshold] * ss
                scores[ctr:ctr + ss] = mv

            # if ctr >= max_prediction_length + max_encoder_length:
            #     new_prediction_data = test_seq[ctr + step - max_prediction_length - max_encoder_length:ctr + step]
            #     new_raw_predictions = best_tft.predict(new_prediction_data, mode="raw", return_x=True)
            #     onepred = new_raw_predictions.output["prediction"][:, :, 3]
            #     onetarget = new_raw_predictions.x["decoder_target"][:, :]
            #     mse_values = torch.mean((onepred - onetarget) ** 2, dim=1)
            #     errors = np.append(errors, mse_values)
            #     mse_quantile = np.quantile(errors, args.quantile)
            #     final_threshold = args.threshold_scale * mse_quantile
            #     thresholds[ctr:ctr + step] = [final_threshold] * step
            #     scores[ctr:ctr + step] = [mse_values] * step



        # determine the results of prediction
        # scores = [0] * max_encoder_length + scores + [0] * max_prediction_length
        # thresholds = [0] * max_encoder_length + thresholds + [0] * max_prediction_length
        preds = [idx for idx in range(len(scores)) if scores[idx] > thresholds[idx]]
        preds = find_middle_values(preds)
        print(scores)
        scores = [tt.item() if tt != 0 else 0 for tt in scores]
        thresholds = [tt.item() if tt != 0 else 0 for tt in thresholds]

        no_CPs += 2
        no_preds += len(preds)
        mark = []
        for j in preds:
            timestamp = ts[j]
            for l in gt_margin:
                if timestamp >= l[0] and timestamp <= l[1]:
                    if l not in mark:
                        mark.append(l)
                    else:
                        no_preds -= 1
                        continue
                    no_TPS += 1
                    delays.append(timestamp - l[2])
        filtered = filtered + [0] * (len(ts) - len(filtered))
        fig = plt.figure()
        fig, ax = plt.subplots(2, figsize=[18, 16], sharex=True)
        ax[0].plot(ts.array, filtered)
        ax[0].axvline(x=ts[startindex], color='green', linestyle='--')
        ax[0].axvline(x=ts[stopindex], color='green', linestyle='--')
        for cp in preds:
            ax[0].axvline(x=ts[cp], color='purple', alpha=0.6)
        ax[1].scatter(ts.array, scores)
        ax[1].scatter(ts.array, thresholds)
        plt.tight_layout()
        plt.savefig(args.path + '/' + tank_sample_id + '.png')

    rec = Evaluation_metrics.recall(no_TPS, no_CPs)
    FAR = Evaluation_metrics.False_Alarm_Rate(no_preds, no_TPS)
    prec = Evaluation_metrics.precision(no_TPS, no_preds)
    f1score = Evaluation_metrics.F1_score(rec, prec)
    f2score = Evaluation_metrics.F2_score(rec, prec)
    # dd = Evaluation_metrics.detection_delay(delays)
    print('recall: ', rec)
    print('false alarm rate: ', FAR)
    print('precision: ', prec)
    print('F1 Score: ', f1score)
    print('F2 Score: ', f2score)
    # print('detection delay: ', dd)

    npz_filename = args.outfile
    np.savez(npz_filename, rec=rec, FAR=FAR, prec=prec, f1score=f1score, f2score=f2score)
