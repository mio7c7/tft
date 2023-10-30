import glob
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from Detector_mmd_real import Detector
from sklearn.metrics.pairwise import pairwise_kernels
import math
import time
import pandas as pd
import os
sys.path.append('./')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='C:/Users/s3912230/Documents/GitHub/tft/pytorch_forecasting/CPD/tankleak.csv', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--bs', type=int, default=150, help='buffer size for ssa')
parser.add_argument('--ws', type=int, default=100, help='window size')
parser.add_argument('--step', type=int, default=10, help='step')
parser.add_argument('--min_requirement', type=int, default=500, help='window size')
parser.add_argument('--memory_size', type=int, default=100, help='memory size per distribution ')
parser.add_argument('--cp_range', type=int, default=5, help='range to determine cp')
parser.add_argument('--forgetting_factor', type=float, default=0.55, help='forgetting_factor')
parser.add_argument('--threshold', type=float, default=4, help='threshold')
parser.add_argument('--quantile', type=float, default=0.975, help='quantile')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='mmd02', help='name of file to save results')
parser.add_argument('--trainsize', type=int, default=4000, help='train size')
parser.add_argument('--validsize', type=int, default=500, help='validtaion size')
parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')

args = parser.parse_args()
# Scale input data to range of -1 to 1
def scale_input(x):
    input_min = 0
    input_max = 1
    return (x - input_min) / (input_max - input_min)

def sliding_window(elements, window_size, step):
    if len(elements) <= window_size:
        return elements
    new = np.empty((0, window_size))
    for i in range(0, len(elements) - window_size + 1, step):
        new = np.vstack((new, elements[i:i+window_size]))
    return new

def maximum_mean_discrepancy(X, Y, kernel='rbf', gamma=0.01):
    K_XX = pairwise_kernels(X, metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y, metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    return mmd

if __name__ == '__main__':
    folder = args.data
    fixed_threshold = 1.5
    error_margin = 864000 # 10 days
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []
    runtime = []
    ignored = []

    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)

    test_sequence = pd.read_csv(args.data)
    test_sequence['period'] = test_sequence['period'].astype(str)
    TRAINSIZE = args.trainsize
    VALIDSIZE = args.validsize
    data = test_sequence[lambda x: x.time_idx <= TRAINSIZE + VALIDSIZE]
    data = data[abs(data['Var_tc_readjusted']) < args.out_threshold]
    tlgrouths = pd.read_csv('C:/Users/s3912230/Documents/GitHub/tft/data_simulation/tankleakage_info.csv',
                            index_col=0).reset_index(drop=True)
    processed_dfs = []
    groups = data.groupby('group_id')
    window_size = 10
    for group_id, group_df in groups:
        group_df = group_df.reset_index(drop=True)
        group_df['time_idx'] = group_df.index
        processed_dfs.append(group_df)
    final_df = pd.concat(processed_dfs, ignore_index=True)
    training_cutoff = 2000 - 48

    for tank_sample_id in list(test_sequence['group_id'].unique()):
        # if tank_sample_id != 'B544_4':
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

        # initialisation for feature extraction module
        reconstructeds = sliding_window(X, args.ws, args.step)
        memory = reconstructeds
        if len(reconstructeds) > args.memory_size:
            random_indices = np.random.choice(len(reconstructeds), size=args.memory_size, replace=False)
            memory = memory[random_indices]
        detector = Detector(args.ws, args)
        detector.addsample2memory(memory, len(memory))

        test_seq = tank_sequence.iloc[training_cutoff:]
        test_seq = test_seq.reset_index(drop=True)
        test_seq['time_idx'] = test_seq.index
        ts = pd.to_datetime(test_seq['Time'])
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
        gt_margin.append((ts[startindex - 10], ts[startindex] + pd.to_timedelta(7, unit='D'), ts[startindex]))
        gt_margin.append((ts[stopindex - 10], ts[stopindex] + pd.to_timedelta(7, unit='D'), ts[stopindex]))

        ctr = 0
        step = args.bs
        scores = [0]*len(ts)
        outliers = []
        preds = []
        filtered = []
        sample = np.empty((0, args.ws))
        collection_period = 1000000000
        detected = False
        thresholds = [0] * len(ts)
        cp_ctr = []
        seq = test_seq['Var_tc_readjusted'].values

        while ctr < seq.shape[0]:
            new = seq[ctr:ctr + step]
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, args.ssa_window-1:]
            reconstructed = updates.sum(axis=0)
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])

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
                    if residual[i1] > threshold_upper or residual[i1] < threshold_lower:
                        outliers.append(ctr + i1)
                        filtered.append(np.mean(filtered[-5:] if len(filtered) > 5 else 0))
                    else:
                        filtered.append(new[i1])

            # detection
            if collection_period > args.min_requirement:
                if ctr == 0:
                    window = np.array(filtered)
                else:
                    window = np.array(filtered[-args.ws - step + 1:])
                if len(window) <= args.ws:
                    break
                window = sliding_window(window, args.ws, args.step)
                for aa in range(len(window)):
                    score = maximum_mean_discrepancy(window[aa].reshape(-1, 1), detector.current_centroid.reshape(-1, 1))
                    scores[ctr + aa * args.step:ctr + aa * args.step + args.step] = [score] * args.step
                    thresholds[ctr + aa * args.step:ctr + aa * args.step + args.step] = [detector.memory_info['threshold']] * args.step
                    if score > detector.memory_info['threshold']:
                        min_dist = 100000
                        detector.N.append(ctr + aa*args.step)
                        collection_period = 0
                        detected = True
                        filtered = filtered[:-len(window) + aa*args.step + 1]
                        detector.newsample = []
                        break
                    else:
                        detector.newsample.append(window[aa])
                # update the rep and threshold for the current distribution
                if collection_period > args.min_requirement:
                    detector.updatememory()
            elif collection_period <= args.min_requirement:
                if len(sample) == 0:
                    window = np.array(filtered[-step + 1:])
                else:
                    window = np.array(filtered[-args.ws - step + 1:])
                if len(window) <= args.ws:
                    break
                window = sliding_window(window, args.ws, args.step)
                if collection_period + step <= args.min_requirement:
                    sample = np.concatenate((sample, window))
                    collection_period += step
                else: #new
                    sample = np.concatenate((sample, window))
                    detector.addsample2memory(sample, len(sample))
                    collection_period = 1000000000
                    sample = np.empty((0, args.ws))
            if detected:
                ctr += aa * args.step + 1
                detected = False
            elif len(seq) - ctr <= args.bs:
                break
            elif len(seq) - ctr <= 2 * args.bs:
                ctr += args.bs
                step = len(seq) - ctr
            else:
                ctr += args.bs

        if len(scores) > len(ts):
            scores = scores[:len(ts)]
            thresholds = thresholds[:len(ts)]

        fig = plt.figure()
        fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        try:
            ax[0].plot(ts, seq)
            for cp in gt_margin:
                ax[0].axvline(x=cp[0], color='green', linestyle='--')
                ax[0].axvline(x=cp[1], color='green', linestyle='--')
            for cp in detector.N:
                ax[0].axvline(x=ts[cp], color='purple', alpha=0.6)
            ax[1].plot(ts, scores)
            ax[1].plot(ts, thresholds)
            # ax[1].plot(ts, mss)
            # ax[2].plot(ts, filtered)
            plt.savefig(args.outfile + '/' + tank_sample_id + '.png')
            plt.close('all')
            del fig
        except:
            print('not able')
        preds = detector.N
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
    np.savez(npz_filename, rec=rec, FAR=FAR, prec=prec, f1score=f1score, f2score=f2score, runtime=sum(runtime)/len(runtime))