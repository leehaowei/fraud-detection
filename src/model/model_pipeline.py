import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


class ModelPipeline:
    def __init__(self, model):
        self.model = model
        self.class_weight = "balanced" if "class_weight" in model.get_params() else None
        self.pipeline = Pipeline(
            [
                ("min_max_scaler", MinMaxScaler()),
                (
                    type(model).__name__,
                    model.set_params(class_weight=self.class_weight)
                    if is_classifier(model)
                    else model,
                ),
            ]
        )

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.pipeline.predict(X_test)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y, set_name="", zero_division=1):
        predictions = self.predict(X)
        y_score = self.predict_proba(X)

        print("Unique classes in y_pred:", np.unique(predictions))
        print("Unique classes in y_encoded:", np.unique(y))

        self.cm = confusion_matrix(y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predictions, average="binary", zero_division=zero_division
        )

        # calculate AUPRC
        auprc = average_precision_score(y, y_score[:, 1])

        # calculate AUCROC
        aucroc = roc_auc_score(y, y_score[:, 1])

        # calculate accuracy
        accuracy = accuracy_score(y, predictions)

        self.metrics_df = pd.DataFrame(
            {
                "Precision": [precision],
                "Recall": [recall],
                "F1 Score": [f1],
                "AUPRC": [auprc],
                "AUCROC": [aucroc],
                "Accuracy": [accuracy],
            },
            index=[set_name],
        )

        return self.metrics_df, self.cm
