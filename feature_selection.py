import logging

from pathlib import Path

import pandas as pd

from pandas import DataFrame

from feature_selection.feature_select import FeatureSelection

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    output_path: Path = Path('data/feature_select/dataset_new.csv')

    df_preprocessed: DataFrame = pd.read_csv('data/preprocess/Breast_Cancer_dataset_preprocessed.csv')

    feature_select = FeatureSelection(df_preprocessed, output_path)

    feature_select.rank_features()  # ranks using RFE; informational only
    feature_select.select_features()  # selects using SFE; this data is saved and used in testing
