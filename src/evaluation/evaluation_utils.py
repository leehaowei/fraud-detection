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


def get_all_scores(y_true, y_pred, alpha=0.5):
    if np.sum(y_true) == np.sum(y_pred) == 0:
        f1 = 1.0
    else:
        f1 = f1_score(y_true, y_pred)

    onset_dev = onset_deviation(y_true, y_pred)
    D = onset_dev / len(y_true)
    score = alpha * (1 - D) + (1 - alpha) * f1

    return D, f1, score


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


def apply_prediction(group, model):
    # Ensure the data is in the correct shape for prediction
    X = group.copy()
    X = X.drop(columns=["motive", "gvkey", "year"])

    # Predict and add results to original data
    group["prediction"] = model.predict(X)
    return group[["motive", "gvkey", "year", "prediction"]]


def get_eva2_score(data, model, alpha=0.5):
    sum_score = 0

    df_prediction = data.groupby("gvkey", group_keys=True).apply(
        apply_prediction, model
    )

    for k in df_prediction["gvkey"].unique():
        filter_ = df_prediction["gvkey"] == k
        temp_df = df_prediction[filter_]

        y_true = temp_df["motive"].values
        y_pred = temp_df["prediction"].values

        eva2_score = combined_score(y_true=y_true, y_pred=y_pred, alpha=alpha)
        # print(f"{k}, score: {eva2_score:.2f}")

        sum_score += eva2_score

    avg_score = sum_score / df_prediction["gvkey"].nunique()
    print("")
    print(f"average score: {avg_score:.2f}")
