import numpy as np
import yaml
from matplotlib import pyplot as plt
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
        model_pipeline = ModelPipeline(model=self.best_model)
        model_pipeline.fit(X, y)

        return model_pipeline

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
        ax.legend()

        # Export the plot to an SVG file if required
        if export_svg:
            plt.savefig(filename, format="svg")

        return fig, ax
