import logging

from pandas import DataFrame, Index
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd


class GradientBoosting:
    def __init__(self, data: DataFrame):
        self._data = data

        # Set the target variable as the index
        self._data.set_index('Status', inplace=True)

        # Encode all categorical features as binary
        self._data_encoded = pd.get_dummies(self._data)

        self._model = GradientBoostingClassifier()

        self._x_train = None
        self._x_test = None
        self._y_train = None
        self._y_test = None

    @property
    def model(self):
        return self._model

    def _fit(self) -> tuple[DataFrame, DataFrame, Index, Index]:
        """ Fit and test a Gradient Boosting classifier via cross validation """
        # Cross-validation
        x_train, x_test, y_train, y_test = train_test_split(self._data_encoded, self._data_encoded.index,
                                                            test_size=0.5)

        # Fit on training DataFrame
        self._model.fit(x_train, y_train)

        return x_train, x_test, y_train, y_test

    def _predict(self, x_test: DataFrame, y_test: Index):
        """ Predict labels against the test set """
        y_predicted = self._model.predict(x_test)

        accuracy = accuracy_score(y_test, y_predicted)
        confusion = confusion_matrix(y_test, y_predicted)

        conf_matrix: DataFrame = pd.DataFrame(confusion,
                                              columns=['Predicted N', 'Predicted Y'])
        conf_matrix.index = ['Actual N', 'Actual Y']

        logging.info(f"Accuracy of Gradient Boosting: {accuracy}")
        logging.info(f"Confusion matrix:\n {conf_matrix.to_markdown(index=False)}")

    def gradient_boosting(self):
        """ Fit the Gradient Boosting model and predict against the test set via cross-validation """
        x_train, x_test, y_train, y_test = self._fit()

        self._predict(x_test, y_test)

        return x_train, x_test, y_train, y_test

    def predict(self):
        return self.gradient_boosting()
