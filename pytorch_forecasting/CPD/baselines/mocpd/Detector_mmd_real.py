import numpy as np
from sklearn.metrics import mean_squared_error
import random
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import pairwise_kernels

def maximum_mean_discrepancy(X, Y, kernel='rbf', gamma=0.01):
    K_XX = pairwise_kernels(X, metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y, metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    return mmd

class Detector():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m
    '''

    def __init__(self, window_size, args):
        self.window_size = window_size
        self.memory = {}  # store existing distributions
        self.memory_info = {} # store the distribution corresponding thresholds
        self.current_centroid = None
        self.N = []
        self.newsample = []
        self.args = args
        self.n_components = 1

    def addsample2memory(self, sample, seen):
        self.memory = {'sample': sample, 'centroid': np.mean(sample, axis=0)}
        self.current_centroid = self.memory['centroid']
        threshold = self.compute_threshold(sample, self.current_centroid, self.args.threshold + 1)
        self.memory_info = {'size': len(sample), 'threshold': threshold, 'seen': seen}

    def resample(self, new_sample):
        org = self.memory['sample']
        old = org
        seen = len(new_sample)
        if self.memory_info['seen'] <= self.args.memory_size+100:
            forgetting_factor = 0.85
            threshold = self.args.threshold + 1
        elif self.memory_info['seen'] <= self.args.memory_size+500:
            forgetting_factor = 0.65
            if self.memory_info['seen'] <= self.args.memory_size+250:
                threshold = self.args.threshold + 1
            else:
                threshold = self.args.threshold + 0.5
        else:
            forgetting_factor = 0.45
            threshold = self.args.threshold
        if len(org) < self.args.memory_size:
            full = self.args.memory_size - len(org)
            org = np.vstack((org, new_sample[:full]))
            new_sample = new_sample[full:]
        for ss in new_sample:
            if random.random() < forgetting_factor:
                org = np.delete(org, 0, axis=0)
                org = np.concatenate((org, np.expand_dims(ss, axis=0)), axis=0)
        sam = org
        self.memory['sample'] = sam
        self.memory_info['threshold'] = self.compute_threshold(old, self.current_centroid,threshold)
        self.memory['centroid'] = np.mean(sam, axis=0)
        self.current_centroid = self.memory['centroid']
        self.memory_info['seen'] += seen

    def updatememory(self):
        self.resample(self.newsample)
        self.newsample = []

    def compute_threshold(self, rep, centroid, threshold):
        MMD = [maximum_mean_discrepancy(rep[i].reshape(-1, 1), centroid.reshape(-1, 1)) for i in range(len(rep))]
        mse_quantile = np.quantile(MMD, self.args.quantile)
        threshold = threshold * mse_quantile
        # threshold = np.mean(MMD) + threshold * np.std(MMD)
        return threshold
