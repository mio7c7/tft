import numpy as np
import pylab
import statistics

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


if __name__ == '__main__':
    y = np.array(
        [1, 1, 1.1, 1, 0.9, 1, 1, 1.1, 1, 0.9, 1, 1.1, 1, 1, 0.9, 1, 1, 1.1, 1, 1, 1, 1, 1.1, 0.9, 1, 1.1, 1, 1, 0.9,
         1, 1.1, 1, 1, 1.1, 1, 0.8, 0.9, 1, 1.2, 0.9, 1, 1, 1.1, 1.2, 1, 1.5, 1, 3, 2, 5, 3, 2, 1, 1, 1, 0.9, 1, 1, 3,
         2.6, 4, 3, 3.2, 2, 1, 1, 0.8, 4, 4, 2, 2.5, 1, 1, 1])

    # Settings: lag = 30, threshold = 5, influence = 0
    lag = 30
    threshold = 5
    influence = 0

    # Run algo with settings from above
    result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)

    # Plot result
    pylab.subplot(211)
    pylab.plot(np.arange(1, len(y) + 1), y)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgFilter"], color="cyan", lw=2)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

    pylab.subplot(212)
    pylab.step(np.arange(1, len(y) + 1), result["signals"], color="red", lw=2)
    pylab.ylim(-1.5, 1.5)
    pylab.show()
