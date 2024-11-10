""" This script runs all available models on the preprocessed SEER dataset """
import logging

import pandas as pd

from pandas import DataFrame

from models.knn import KNearestNeighbors
from models.naive_bayes import NaiveBayes
from models.random_forest import RandomForest
from models.gradient_boosting import GradientBoosting
from models.neural_network import NeuralNetwork
from models.hyperparameter_tuning import HyperparameterSearch


class Modeling:
    def __init__(self):
        self._df_postprocessed: DataFrame = pd.read_csv('data/feature_select/dataset_new.csv')

        # Model instances
        self._knn = KNearestNeighbors(self._df_postprocessed.copy(), k=3)
        self._nb = NaiveBayes(self._df_postprocessed.copy())
        self._rf = RandomForest(self._df_postprocessed.copy())
        self._gb = GradientBoosting(self._df_postprocessed.copy())
        self._nn = NeuralNetwork(self._df_postprocessed.copy())

        # Training and test data
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None

    def knn(self):
        # custom implementation
        self._knn.predict()

    def random_forest(self):
        # scikit-learn wrapper
        self._x_train, self._x_test, self._y_train, self._y_test = self._rf.predict()

    def naive_bayes(self):
        # scikit-learn wrapper
        self._x_train, self._x_test, self._y_train, self._y_test = self._nb.predict()

    def gradient_boosting(self):
        # scikit-learn wrapper
        self._x_train, self._x_test, self._y_train, self._y_test = self._gb.predict()

    def neural_network(self):
        # scikit-learn wrapper
        self._x_train, self._x_test, self._y_train, self._y_test = self._nn.predict()

    def hyperparameter_search_rf(self):
        param_search: HyperparameterSearch = HyperparameterSearch(self._rf.model, HyperparameterSearch.params_rf)

        param_search.grid_search(self._x_train, self._y_train, self._x_test, self._y_test)

    def hyperparameter_search_gb(self):
        param_search: HyperparameterSearch = HyperparameterSearch(self._gb.model, HyperparameterSearch.params_gb)

        param_search.grid_search(self._x_train, self._y_train, self._x_test, self._y_test)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    modeling: Modeling = Modeling()

    modeling.knn()
    modeling.random_forest()
    modeling.naive_bayes()
    modeling.gradient_boosting()
    modeling.neural_network()

    modeling.hyperparameter_search_rf()
    modeling.hyperparameter_search_gb()
