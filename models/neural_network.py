""" Fit a NN to the test data. scikit-learn is not optimal or performant for Neural Networks,
 and other libraries (such as pytorch) are more suitable for production usage.

 However, its implementation of NNs follows the same
    design as all the other models and as such is super convenient to test with here.
"""
import logging

import numpy as np
import pandas as pd

from pandas import DataFrame, Index
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix


class NeuralNetwork:
    def __init__(self, data: DataFrame):
        self._data = data

        # Set the target variable as the index
        self._data.set_index('Status', inplace=True)

        # Encode all categorical features as binary
        self._data_encoded = pd.get_dummies(self._data)

        self._model = MLPRegressor(max_iter=2000)

    @property
    def model(self):
        return self._model

    def _fit(self):
        """ Fit and test a NN via cross validation """
        # Cross-validation
        x_train, x_test, y_train, y_test = train_test_split(self._data_encoded, self._data_encoded.index,
                                                            test_size=0.5, random_state=20)

        # Fit on training DataFrame
        self._model.fit(x_train, y_train)

        return x_train, x_test, y_train, y_test

    def _predict(self, x_test: DataFrame, y_test: Index):
        """ Predict labels against the test set """
        y_predicted = self._model.predict(x_test)

        # Convert to binary labels
        y_predicted_rounded = np.round(y_predicted)

        accuracy = accuracy_score(y_test, y_predicted_rounded)
        confusion = confusion_matrix(y_test, y_predicted_rounded)

        conf_matrix: DataFrame = pd.DataFrame(confusion,
                                              columns=['Predicted N', 'Predicted Y'])
        conf_matrix.index = ['Actual N', 'Actual Y']

        logging.info(f"Accuracy of Neural Network: {accuracy}")
        logging.info(f"Confusion matrix:\n {conf_matrix.to_markdown(index=False)}")

    def neural_network(self):
        """ Fit a NN to the test data. scikit-learn is not optimal for this model, but is simple to test with """
        x_train, x_test, y_train, y_test = self._fit()

        self._predict(x_test, y_test)

        return x_train, x_test, y_train, y_test

    def predict(self):
        return self.neural_network()
