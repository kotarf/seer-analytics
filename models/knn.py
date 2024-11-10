""" This is an implementation of a K Nearest Neighbor classifier, using pandas for sorting, indexing etc.

No normalization is done on the input features, nor is cross-validation used.
This happens to work okay with the SEER dataset, but in general this code would work well on arbitrary datasets.
"""
import logging

from typing import Optional

import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from numpy import ndarray

TestPoint = ndarray
Distances = ndarray
InputData = DataFrame
Label = str


class KNearestNeighbors:
    col_distance = 'distance'

    split_factor: float = .8

    def __init__(self, data: InputData, k: int = 3):
        assert 'Status' in data.columns  # label

        self._data: InputData = data
        self._k = k

        self._df_train: Optional[DataFrame] = None
        self._df_test: Optional[DataFrame] = None

        self._data.set_index('Status', inplace=True)

    def _split_training(self):
        # Split into training and test data by an 80/20 proportion. No cross-validation is used here.
        training_ix: ndarray = np.random.permutation(len(self._data))
        training_cutoff: int = int(len(self._data) * KNearestNeighbors.split_factor)

        self._df_train = self._data.iloc[training_ix[:training_cutoff]].copy()
        self._df_test = self._data.iloc[training_ix[training_cutoff:]].copy()

        logging.info(f"Size of training set: {len(self._df_train)}")
        logging.info(f"Size of test set: {len(self._df_test)}")

    def _calculate_distances(self, test_point: TestPoint) -> Distances:
        # Euclidean distance metric
        distances: Distances = np.linalg.norm(test_point - self._df_train, axis=1)

        return distances

    def _sort_distances(self):
        # Sorts all data in the training set by distance to the test point
        self._df_train.sort_values(ascending=True, by=KNearestNeighbors.col_distance, inplace=True)

    def _nearest_neighbors(self, test_point: TestPoint) -> DataFrame:
        self._df_train.loc[:, KNearestNeighbors.col_distance] = self._calculate_distances(test_point)

        self._sort_distances()

        nearest_neighbor_vals: DataFrame = self._df_train.head(self._k)

        return nearest_neighbor_vals

    def _classify(self, test_point: TestPoint) -> Label:
        # Compute the nearest neighbors to the test point
        nearest_neighbors: DataFrame = self._nearest_neighbors(test_point)

        # Select a label by majority vote
        nearest_neighbor_labels: Series = nearest_neighbors.index.value_counts()
        label_majority: Label = str(nearest_neighbor_labels.idxmax())

        # Drops the temporary 'distance' column
        self._df_train.drop(labels=[KNearestNeighbors.col_distance], axis=1, inplace=True)

        return label_majority

    @staticmethod
    def _performance(df_predicted: DataFrame) -> Optional[float]:
        """ Computes the performance of the KNN classifier """
        df_predicted.reset_index(names='Original Label', inplace=True)

        # Encode to binary
        df_predicted = pd.get_dummies(df_predicted, columns=['Original Label', 'Predicted Label'])

        if 'Original Label_0' not in df_predicted.columns:
            df_predicted.loc[:, 'Original Label_0'] = False
        if 'Original Label_1' not in df_predicted.columns:
            df_predicted.loc[:, 'Original Label_1'] = False
        if 'Predicted Label_0' not in df_predicted.columns:
            df_predicted.loc[:, 'Predicted Label_0'] = False
        if 'Predicted Label_1' not in df_predicted.columns:
            df_predicted.loc[:, 'Predicted Label_1'] = False

        true_negative: int = df_predicted[(df_predicted['Original Label_0'] == True)
                                          & (df_predicted['Predicted Label_0'] == True)].shape[0]
        false_negative: int = df_predicted[(df_predicted['Original Label_0'] == True)
                                           & (df_predicted['Predicted Label_0'] == False)].shape[0]

        true_positive: int = df_predicted[(df_predicted['Original Label_1'] == True)
                                          & (df_predicted['Predicted Label_1'] == True)].shape[0]
        false_positive: int = df_predicted[(df_predicted['Original Label_1'] == True)
                                           & (df_predicted['Predicted Label_1'] == False)].shape[0]

        total_count: int = true_negative + true_positive + false_negative + false_positive

        if total_count == 0:
            logging.warning("Testing set was insufficient to compute accuracy")

            return None

        # Accuracy
        accuracy_val: float = (true_negative + true_positive) / total_count

        # Confusion matrix
        conf_matrix: DataFrame = pd.DataFrame([(true_negative, false_negative), (false_positive, true_positive)],
                                              columns=['Predicted N', 'Predicted Y'])
        conf_matrix.index = ['Actual N', 'Actual Y']

        logging.info(f"Accuracy of KNN: {accuracy_val}")
        logging.info(conf_matrix.to_markdown())

        return accuracy_val

    def classify_test_point(self, test_point: TestPoint):
        # Classify a single test point. Good for debugging.
        self._split_training()

        return self._classify(test_point)

    def classify_all(self):
        # Split the data into training/test, and then classify all points in the test data
        self._split_training()

        # Classify every test point
        classified_points: Series = self._df_test.apply(self._classify, axis=1)

        return classified_points

    def knn(self):
        # Classify every test point
        classified_points: Series = self.classify_all()

        # Join against the original test DataFrame
        self._df_test.loc[:, 'Predicted Label'] = classified_points

        # Compute performance
        self._performance(self._df_test)

    def predict(self):
        return self.knn()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # This is a test dataset and instance
    dataset = pd.DataFrame([
        (1.0, 1.0, 0),
        (1.0, 1.0, 0),
        (1.0, 1.0, 0),
        (1.0, 1.0, 0),
        (1.0, 1.0, 0),
        (1.0, 1.0, 0),
        (1.5, 1.5, 0),
        (1.5, 1.5, 0),
        (1.2, 1.2, 0),
        (1.2, 1.2, 0),
        (1.2, 1.2, 0),
        (4.2, 4.2, 1),
        (4.2, 4.2, 1),
        (4.2, 4.2, 1),
        (4.0, 4.0, 1),
        (4.0, 4.0, 1),
        (4.0, 4.0, 1),
        (4.0, 4.0, 1),
        (5.0, 5.0, 1),
        (5.0, 5.0, 1),
        (5.0, 5.0, 1),
        (5.0, 5.0, 1),
        (5.0, 5.0, 1),
        (5.0, 5.0, 1),
    ], columns=['feature_a', 'feature_b', 'Status'])

    test_instance = [1.0, 1.0]

    knn: KNearestNeighbors = KNearestNeighbors(dataset, 3)

    print("Predicted Label:", knn.classify_test_point(test_instance))  # noqa

    knn.knn()
