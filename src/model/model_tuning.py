from joblib import dump
import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from .model_pipeline import ModelPipeline


class ModelTuning:
    def __init__(self, model, param_grid, cv_splits=5, random_state=None):
        self.model = model
        self.param_grid = param_grid
        self.cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        self.best_model = None

    def fit(self, X, y):
        grid = GridSearchCV(self.model, self.param_grid, cv=self.cv)
        grid.fit(X, y)

        # Check the best parameters
        print("Best parameters found by grid search are:", grid.best_params_)

        # Get the best estimator
        self.best_model = grid.best_estimator_

        # Fit the best model to the data
        self.model_pipeline = ModelPipeline(model=self.best_model)
        self.model_pipeline.fit(X, y)

        return self.model_pipeline

    def export_model_pipeline(self, file_path):
        dump(self.model_pipeline, file_path)

    def evaluate(self, X, y, set_name=""):
        if self.best_model is None:
            raise Exception("You must call fit() before evaluate().")

        model_pipeline = ModelPipeline(model=self.best_model)
        model_pipeline.fit(X, y)
        metrics_df, cm = model_pipeline.evaluate(X, y, set_name=set_name)
        return metrics_df, cm

    def get_model_params(self):
        if self.best_model is None:
            raise Exception("You must call fit() before get_model_params().")

        return self.best_model.get_params()

    def get_model_coefs(self):
        if self.best_model is None:
            raise Exception("You must call fit() before get_model_coefs().")

        return self.best_model.coef_, self.best_model.intercept_

    def export_model_params(self, file_path: str):
        model_params_dict = self.get_model_params()

        with open(file_path, "w") as file:
            yaml.dump(model_params_dict, file)

    @staticmethod
    def plot_comparison(
        models_eval_dict, export_svg=False, filename="model_comparison.svg"
    ):
        """
        Plot comparison of different models

        :param models_eval_dict: a dictionary that contains model names as keys and corresponding evaluation metrics dataframes as values
        :param export_svg: boolean, if True the plot will be saved to an SVG file
        :param filename: string, name of the file where the plot will be saved (only if export_svg is True)
        :return: the matplotlib figure and axes objects
        """
        metrics = ["Precision", "Recall", "F1 Score", "AUPRC", "AUCROC", "Accuracy"]
        model_names = list(models_eval_dict.keys())
        bar_width = 0.2
        r = np.arange(len(metrics))  # the label locations

        fig, ax = plt.subplots(figsize=(10, 7))

        # Define color scheme
        colors = ["#1f77b4", "#17becf", "#aec7e8"]  # shades of blue

        # Create bars for each model
        for i, model in enumerate(model_names):
            model_scores = [
                models_eval_dict[model].loc["KFold", metric] for metric in metrics
            ]
            r_model = [x + bar_width * i for x in r]
            ax.barh(
                r_model, model_scores, height=bar_width, color=colors[i], label=model
            )

        ax.set_xlabel("Score")
        ax.set_xlim([0, 1])
        ax.set_ylabel("Metric")
        ax.set_yticks([r + bar_width for r in range(len(metrics))])
        ax.set_yticklabels(metrics)

        # Set the location of the legend to the bottom left of the plot
        ax.legend(loc="upper right", bbox_to_anchor=(0.0, -0.10))

        # Export the plot to an SVG file if required
        if export_svg:
            plt.savefig(filename, format="svg", bbox_inches="tight")

        return fig, ax

    @staticmethod
    def plot_roc_curves(X, y, models, title="Receiver Operating Characteristic"):
        """
        Plots ROC curves for models

        :param X: feature dataset
        :param y: target variable
        :param models: a dictionary containing model names as keys and trained model instances as values
        :param title: title of the plot
        :return: the matplotlib figure and axes objects
        """
        plt.figure(figsize=(10, 8))
        colors = ["#1f77b4", "#17becf", "#aec7e8"]  # shades of blue
        lw = 2

        for i, (model_name, model) in enumerate(models.items()):
            probs = model.predict_proba(X)
            preds = probs[:, 1]
            fpr, tpr, threshold = roc_curve(y, preds)
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                color=colors[i],
                lw=lw,
                label="%s ROC curve (area = %0.2f)" % (model_name, roc_auc),
            )

        # Include the line for a random classifier in the legend by giving it a label
        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=lw,
            linestyle="--",
            label="Random Classifier (area = 0.50)",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")

        return plt.figure, plt.gca()

    @staticmethod
    def plot_pr_curves(X, y, models, title="Precision-Recall Curve"):
        """
        Plots Precision-Recall curves for models

        :param X: feature dataset
        :param y: target variable
        :param models: a dictionary containing model names as keys and trained model instances as values
        :param title: title of the plot
        :return: the matplotlib figure and axes objects
        """
        plt.figure(figsize=(10, 8))
        colors = ["#1f77b4", "#17becf", "#aec7e8"]  # shades of blue
        lw = 2

        for i, (model_name, model) in enumerate(models.items()):
            probs = model.predict_proba(X)
            preds = probs[:, 1]
            precision, recall, _ = precision_recall_curve(y, preds)
            pr_auc = auc(recall, precision)

            plt.plot(
                recall,
                precision,
                color=colors[i],
                lw=lw,
                label="%s PR curve (area = %0.2f)" % (model_name, pr_auc),
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="lower right")

        return plt.figure, plt.gca()
