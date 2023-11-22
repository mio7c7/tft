import numpy as np
import pylab
import statistics
from math import *
import matplotlib.pyplot as plt
from scipy.stats import norm

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def PTF_ONE(stream, k):
    padding_value = 0
    padded_stream = [padding_value] * (k - 1) + stream + [padding_value] * (k - 1)
    transformed = [0]*len(stream)
    for i in range(k, len(padded_stream)-k):
        left_win = [padded_stream[i] - j for j in padded_stream[i-k:i]]
        right_win = [padded_stream[i] - j for j in padded_stream[i:i+k]]
        transformed[i-k] = (max(left_win) + max(right_win))/2
    return transformed

def PTF_TWO(stream, k):
    padding_value = 0
    padded_stream = [padding_value] * (k - 1) + stream + [padding_value] * (k - 1)
    transformed = [0]*len(stream)
    for i in range(k, len(padded_stream)-k):
        left_win = [padded_stream[i] - j for j in padded_stream[i - k:i]]
        right_win = [padded_stream[i] - j for j in padded_stream[i:i + k]]
        transformed[i-k] = (statistics.mean(left_win) + statistics.mean(right_win))/2
    return transformed

def MAM(transformed, seg_size):
    """
    Here, a segment with pre-defined number of points is created around each point. Statistics of the segment mainly
    mean, RMS and standard deviation are computed, and threshold is calculated based on these parameters.
    """
    padding_value = 0
    padded_stream = [padding_value] * (seg_size - 1) + transformed
    means, stds, rmss = [0]*len(transformed), [0]*len(transformed), [0]*len(transformed)
    for i in range(seg_size, len(padded_stream)):
        left_win = padded_stream[i - seg_size:i]
        means[i - seg_size] = statistics.mean(left_win)
        stds[i - seg_size] = statistics.stdev(left_win)
        rmss[i - seg_size] = statistics.mean(left_win) + 4 * statistics.stdev(left_win)
        # rmss[i - seg_size] = np.sqrt(sum([(x - statistics.mean(left_win))**2 for x in left_win]))
    return means, stds, rmss

def NAB(stream, long_ws, short_ws):
    padding_value = 0
    padded_stream = [padding_value] * (long_ws - 1) + stream
    anomaly_scores = [0] * len(stream)
    for i in range(long_ws, len(padded_stream)):
        long_win = padded_stream[i - long_ws:i]
        short_win = padded_stream[i - short_ws:i]
        mean_l = statistics.mean(long_win)
        var_l = statistics.stdev(long_win)
        mean_s = statistics.mean(short_win)
        anomaly_L = 1 - norm.sf(mean_s, loc=mean_l, scale=var_l)
        anomaly_scores[i - long_ws] = anomaly_L
    return anomaly_scores

class HDDM_A():
    def __init__(self, drift_confidence=0.001, warning_confidence=0.005):
        self.n_min = 0
        self.c_min = 0
        self.total_n = 0
        self.total_c = 0
        self.n_max = 0
        self.c_max = 0
        self.n_estimation = 0
        self.c_estimation = 0
        self.in_concept_change = None
        self.in_warning_zone = None
        self.estimation = None
        self.delay = None

        self.drift_confidence = drift_confidence #alpha
        self.warning_confidence = warning_confidence

    def add_element(self, error):
        self.total_n += 1
        self.total_c += error
        if self.n_min == 0:
            self.n_min = self.total_n
            self.c_min = self.total_c
        if self.n_max == 0:
            self.n_max = self.total_n
            self.c_max = self.total_c

        cota = sqrt(1.0 / (2 * self.n_min) * log(1.0 / self.drift_confidence))
        cota1 = sqrt(1.0 / (2 * self.total_n) * log(1.0 / self.drift_confidence))

        if self.c_min / self.n_min + cota >= self.total_c / self.total_n + cota1:
            self.c_min = self.total_c
            self.n_min = self.total_n

        cota = sqrt(1.0 / (2 * self.n_max) * log(1.0 / self.drift_confidence))
        if self.c_max / self.n_max - cota <= self.total_c / self.total_n - cota1:
            self.c_max = self.total_c
            self.n_max = self.total_n

        if self._mean_incr(self.c_min, self.n_min, self.total_c, self.total_n, self.drift_confidence):
            self.n_estimation = self.total_n - self.n_min
            self.c_estimation = self.total_c - self.c_min
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0
            self.in_concept_change = True
            self.in_warning_zone = False
        elif self._mean_incr(self.c_min, self.n_min, self.total_c, self.total_n, self.warning_confidence):
            self.in_concept_change = False
            self.in_warning_zone = True
        else:
            self.in_concept_change = False
            self.in_warning_zone = False
        self._update_estimations()

    def _mean_incr(self, c_min, n_min, total_c, total_n, confidence):
        if n_min == total_n:
            return False

        m = (total_n - n_min) / n_min * (1.0 / total_n)
        cota = sqrt(m / 2 * log(2.0 / confidence))
        return total_c / total_n - c_min / n_min >= cota

    def _update_estimations(self):
        """ update_estimations

        Update the length estimation and delay.

        """
        if self.total_n >= self.n_estimation:
            self.c_estimation = self.n_estimation = 0
            self.estimation = self.total_c / self.total_n
            self.delay = self.total_n
        else:
            self.estimation = self.c_estimation / self.n_estimation
            self.delay = self.n_estimation

    def detected_change(self):
        return self.in_concept_change

    def detected_warning_zone(self):
        return self.in_warning_zone

if __name__ == '__main__':
    data_dict = np.load('errors.npy', allow_pickle=True).item()
    for key, value in data_dict.items():
        transformed = PTF_TWO(value, k=48)
        means, stds, rmss = MAM(transformed, seg_size=48)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # Plot on the first subplot
        ax1.plot(value, label='errors', color='blue')
        ax1.set_title('Plot 1')
        ax1.legend()

        # Plot on the second subplot
        ax2.plot(transformed, label='trasnformed', color='red')
        ax2.plot(means, label='mean', color='blue')
        ax2.plot(stds, label='standard deviation', color='green')
        ax2.plot(rmss, label='threshold', color='orange')
        mark = []
        for index, value in enumerate(transformed):
            if value > rmss[index]:
                mark.append(index)
        trans = [transformed[i] for i in mark]
        ax2.plot(mark, trans, marker="o", ls="", ms=3, color='black')
        ax2.set_title('Plot 2')
        ax2.legend()

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.savefig(key+'.png')

    # hddm_a = HDDM_A()
    # data_stream = np.random.randint(2, size=2000)
    # for i in range(999, 1500):
    #     data_stream[i] = 0
    # for i in range(2000):
    #     hddm_a.add_element(data_stream[i])
    #     if hddm_a.detected_warning_zone():
    #         print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    #     if hddm_a.detected_change():
    #         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

    # y = np.array(
    #     [1, 1, 1.1, 1, 0.9, 1, 1, 1.1, 1, 0.9, 1, 1.1, 1, 1, 0.9, 1, 1, 1.1, 1, 1, 1, 1, 1.1, 0.9, 1, 1.1, 1, 1, 0.9,
    #      1, 1.1, 1, 1, 1.1, 1, 0.8, 0.9, 1, 1.2, 0.9, 1, 1, 1.1, 1.2, 1, 1.5, 1, 3, 2, 5, 3, 2, 1, 1, 1, 0.9, 1, 1, 3,
    #      2.6, 4, 3, 3.2, 2, 1, 1, 0.8, 4, 4, 2, 2.5, 1, 1, 1])
    #
    # # Settings: lag = 30, threshold = 5, influence = 0
    # lag = 30
    # threshold = 5
    # influence = 0
    #
    # # Run algo with settings from above
    # result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)
    #
    # # Plot result
    # pylab.subplot(211)
    # pylab.plot(np.arange(1, len(y) + 1), y)
    #
    # pylab.plot(np.arange(1, len(y) + 1),
    #            result["avgFilter"], color="cyan", lw=2)
    #
    # pylab.plot(np.arange(1, len(y) + 1),
    #            result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)
    #
    # pylab.plot(np.arange(1, len(y) + 1),
    #            result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)
    #
    # pylab.subplot(212)
    # pylab.step(np.arange(1, len(y) + 1), result["signals"], color="red", lw=2)
    # pylab.ylim(-1.5, 1.5)
    # pylab.show()
