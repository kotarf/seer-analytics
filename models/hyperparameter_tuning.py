""" Performs Hyperparameter tuning on the RF and Gradient Boosting models """
import logging

from pandas import DataFrame, Index
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score


class HyperparameterSearch:
    params_rf: dict = {
        'max_depth': [None, 10, 20, 30, 50, 75, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
        'n_estimators': [50, 100, 150],
    }
    params_gb: dict = {
        'learning_rate': [.01, .1, .25, .33],
        'max_depth': [None, 10, 20, 30, 50, 75, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
        'n_estimators': [50, 100, 150],
    }

    def __init__(self, model, params: dict):
        self._model = model
        self._search_params = params

    def grid_search(self, x_train: DataFrame, y_train: DataFrame, x_test: Index, y_test: Index):
        grid_search = RandomizedSearchCV(self._model, self._search_params, n_iter=20)

        grid_search.fit(x_train, y_train)

        # Evaluate both models
        best_estimator = grid_search.best_estimator_

        y_pred_grid = best_estimator.predict(x_test)

        df_tuning: DataFrame = DataFrame.from_dict([grid_search.best_params_]) # noqa

        logging.info(f"Best Parameters and Accuracy for {self._model.__module__}:")
        logging.info(df_tuning.to_markdown(index=False))
        logging.info(accuracy_score(y_test, y_pred_grid))
