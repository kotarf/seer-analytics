import logging

from pathlib import Path
from typing import Optional

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.svm import SVR

from pandas import DataFrame


class FeatureSelection:
    col_target = "Status"  # e.g. Alive

    def __init__(self, df_cancer: DataFrame, output_path: Path):
        self._df = df_cancer
        self._output_path = output_path
        self._df_target = self._df[self.col_target]
        self._df_features = self._df.drop(labels=self.col_target, axis=1)
        self._features_selected: Optional[list[str]] = None
        self._model = LogisticRegression(max_iter=10000)

    def _rank_features_rfe(self):
        """ Performs feature ranking via recursive feature elimination to compare to SFE.
         This is ONLY used to rank the features for discussion; model performance is better with the SFE results """
        selector_rfe = RFE(SVR(kernel="linear"), n_features_to_select=1).fit(self._df_features, self._df_target)

        rank_values = selector_rfe.ranking_  # noqa

        df_ranks: DataFrame = (DataFrame({'Ranked Feature': self._df_features.columns, 'Rank': rank_values})
                               .sort_values(by='Rank'))

        logging.info('Ranked Features via RFE (test only):')
        logging.info(df_ranks.to_markdown(index=False))

    def _select_features(self):
        """ Perform Sequential Feature Selection using a Logistic Regression via scikit-learn.
         Note that due to library limitations, rankings are not outputted. """
        self._selector = SequentialFeatureSelector(self._model, n_features_to_select=5)

        self._x_new = self._selector.fit_transform(self._df_features, self._df_target)

        # List the n selected features
        self._features_selected = self._selector.get_feature_names_out().tolist()

        logging.info('Selected Features via SFE:')
        logging.info(self._features_selected)

    def _output_new_features(self):
        """ Output selected features with some additional normalization """
        data_new: DataFrame = self._df[self._features_selected + ['Status']]

        # Convert continuous data to categorical in order to use a Categorical Naive Bayes, C.45 Decision Tree etc
        # This was deferred to get more accurate feature selection
        if 'Survival Months' in data_new.columns:
            data_new.loc[:, 'Survival Months'] = pd.qcut(data_new['Survival Months'], 8, labels=False)
        if 'Tumour' in data_new.columns:
            data_new.loc[:, 'Tumor Size'] = pd.qcut(data_new['Tumor Size'], 4, labels=False)
        if 'Regional Node Examined' in data_new.columns:
            data_new.loc[:, 'Regional Node Examined'] = pd.qcut(data_new['Regional Node Examined'], 3, labels=False,
                                                                duplicates='drop')
        if 'Regional Node Positive' in data_new.columns:
            data_new.loc[:, 'Regional Node Positive'] = pd.qcut(data_new['Regional Node Positive'], 3, labels=False,
                                                                duplicates='drop')

        data_new.to_csv(self._output_path, index=False)

    def rank_features(self):
        """ Perform feature ranking using RFE. This is only used to rank features.
         The models were all tested with features selected through SFE."""
        self._rank_features_rfe()

    def select_features(self):
        """ Perform feature selection and output selected features with some additional normalization """
        self._select_features()

        self._output_new_features()
