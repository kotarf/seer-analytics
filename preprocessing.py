""" Preprocesses the cancer dataset CSV. While all this could be done in Excel, it's quicker to do via Python & pandas """
from pathlib import Path
from typing import Optional

import pandas as pd

from pandas import DataFrame


class Preprocessor:
    map_marital_status: dict[str, int] = {
        'Single': 1,
        'Married': 2,
        'Divorced': 3,
        'Widowed': 4,
    }
    map_race: dict[str, int] = {
        'White': 1,
        'Black': 2,
        'Other': 3,
    }
    map_t_stage: dict[str, int] = {
        'T1': 1,
        'T2': 2,
        'T3': 3,
        'T4': 4,
    }
    map_n_stage: dict[str, int] = {
        'N1': 1,
        'N2': 2,
        'N3': 3,
        'N4': 4,
    }
    map_differentiate: dict[str, int] = {
        'Poorly differentiated': 1,
        'Moderately differentiated': 2,
        'Well differentiated': 3,
    }
    map_6th_stage: dict[str, int] = {
        'IA': 1,
        'IB': 2,
        'IIA': 3,
        'IIB': 4,
        'IIIA': 5,
        'IIIB': 6,
        'IIIC': 7,
    }
    map_a_stage: dict[str, int] = {
        'Regional': 0,
        'Distant': 1,
    }
    map_estrogen_status: dict[str, int] = {
        'Positive': 0,
        'Negative': 1,
    }
    map_progesterone_status: dict[str, int] = {
        'Positive': 0,
        'Negative': 1,
    }
    map_vital_status: dict[str, int] = {
        'Alive': 0,
        'Dead': 1
    }

    def __init__(self, input_path: Path, output_path: Path):
        self._data = pd.read_csv(input_path)
        self._output_path = output_path
        self._data_preprocessed: Optional[DataFrame] = None

    def _clean_columns(self):
        # Replace whitespace in all columns
        self._data.columns = [col.strip() for col in self._data.columns]

        # Rename column with typo
        self._data.rename(columns={'Reginol Node Positive': 'Regional Node Positive'}, inplace=True)

    def _standardize_values(self):
        # Replaces string data with numerical categories
        self._data['Race'] = self._data['Race'].map(Preprocessor.map_race)
        self._data['Marital Status'] = self._data['Marital Status'].map(Preprocessor.map_marital_status)
        self._data['T Stage'] = self._data['T Stage'].map(Preprocessor.map_t_stage)
        self._data['N Stage'] = self._data['N Stage'].map(Preprocessor.map_n_stage)
        self._data['6th Stage'] = self._data['6th Stage'].map(Preprocessor.map_6th_stage)
        self._data['differentiate'] = self._data['differentiate'].map(Preprocessor.map_differentiate)

        self._data['A Stage'] = self._data['A Stage'].map(Preprocessor.map_a_stage)
        self._data['Estrogen Status'] = self._data['Estrogen Status'].map(Preprocessor.map_estrogen_status)
        self._data['Progesterone Status'] = self._data['Progesterone Status'].map(Preprocessor.map_progesterone_status)
        self._data['Status'] = self._data['Status'].map(Preprocessor.map_vital_status)

    def _drop_missing(self):
        # This drops any rows with null values. This is not ideal, but simple to implement; imputation would best
        # be done (but needs a framework like scikit-learn)
        self._data.dropna(axis='rows', how='any', inplace=True)

    def _force_types(self):
        # Some columns must be forced to integer
        self._data['differentiate'] = self._data['differentiate'].astype(int)

    def _output_new_data(self):
        # Outputs the preprocessed data
        self._data.to_csv(self._output_path, index=False)

    def preprocess_data(self):
        """ Runs all preprocessing steps, including dropping nulls and standardizing values """
        self._clean_columns()
        self._standardize_values()
        self._drop_missing()
        self._force_types()

        self._output_new_data()


if __name__ == '__main__':
    input_path: Path = Path('data/reference/Breast_Cancer_dataset.csv')
    output_path: Path = Path('data/preprocess/Breast_Cancer_dataset_preprocessed.csv')

    preprocessor: Preprocessor = Preprocessor(input_path, output_path)

    preprocessor.preprocess_data()
