import numpy as np
from sklearn.metrics import f1_score


def combined_score(y_true, y_pred, alpha=0.5):
    if np.sum(y_true) == np.sum(y_pred) == 0:
        f1 = 1.0
    else:
        f1 = f1_score(y_true, y_pred)

    onset_dev = onset_deviation(y_true, y_pred)
    score = alpha * (1 - onset_dev / len(y_true)) + (1 - alpha) * f1

    return score


def onset_deviation(y_true, y_pred):
    try:
        onset_true = np.where(y_true == 1)[0][0]
    except:
        onset_true = len(y_true)

    try:
        onset_pred = np.where(y_pred == 1)[0][0]
    except:
        onset_pred = len(y_pred)

    return np.abs(onset_true - onset_pred)
