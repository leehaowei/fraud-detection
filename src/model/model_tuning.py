import yaml
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
