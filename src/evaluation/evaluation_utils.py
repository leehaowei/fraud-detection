import numpy as np
from sklearn.metrics import f1_score


def combined_score(y_true, y_pred):
    onset_dev = onset_deviation_normalized(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return (onset_dev + f1) / 2


def onset_deviation_normalized(y_true, y_pred):
    onsets_true = np.where(np.diff(y_true) == 1)[0] + 1
    onsets_pred = np.where(np.diff(y_pred) == 1)[0] + 1

    if len(onsets_pred) == 0:
        # If no fraud is predicted, return 0
        return 0
    else:
        total_deviation = 0
        for pred in onsets_pred:
            # Find the closest true onset for each predicted onset
            closest_true_onset = min(onsets_true, key=lambda x: abs(x - pred))
            total_deviation += abs(pred - closest_true_onset) / len(y_true)

        # Take the average of the deviations
        avg_deviation = total_deviation / len(onsets_pred)

        # Return the inverted and normalized score
        return 1 - avg_deviation
