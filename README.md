This is a model testing project that takes as input data from the SEER database. This data is used to predict
survivability outcomes of breast cancer patients. There are three distinct scripts that should be run in order:

* preprocessing.py
* feature_selection.py
* modeling.py

The preprocessing script will perform data preprocessing on the given SEER dataset.

Feature selection will output the top 5 features to be used.

Finally, the modeling script will test a series of models (mostly using scikit-learn) against the dataset to predict
survivability outcomes.