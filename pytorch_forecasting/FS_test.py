import os
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

os.chdir("../../..")
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
max_prediction_length = 2*24 #the goal is to make a one-day forecast 48
max_encoder_length = 7*2*24
group = 0# a week 336
folder = 'C:/Users/Administrator/Documents/GitHub/tft/data_simulation/*_Tank.csv'
dfs = []
for i in glob.glob(folder):
    data = pd.read_csv(i, index_col=0).reset_index(drop=True)
    data = data.iloc[:2000]
    data['group_id'] = group
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

