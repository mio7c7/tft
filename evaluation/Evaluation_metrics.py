def recall(no_TP, no_CPS):
    return no_TP/no_CPS

def False_Alarm_Rate(no_pred, no_TP):
    return (no_pred - no_TP)/no_pred

def precision(no_TP, no_pred):
    return no_TP/no_pred

def F1_score(recall, precision):
    return 2 * recall * precision / (recall + precision)

def F2_score(recall, precision):
    return 5 * recall * precision / (recall + 4*precision)

def detection_delay(delays):
    # delays should store all the detection delays for all the correctly detected change points
    return sum(delays)/len(delays)