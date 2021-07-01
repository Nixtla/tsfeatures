#!/usr/bin/env python
# coding: utf-8

from tsfeatures import tsfeatures
from tsfeatures.m4_data import prepare_m4_data
from tsfeatures.utils import FREQS

def test_pipeline():
    def calculate_features_m4(dataset_name, directory, num_obs=1000000):
        _, y_train_df, _, _ = prepare_m4_data(dataset_name=dataset_name,
                                              directory = directory,
                                              num_obs=num_obs)

        freq = FREQS[dataset_name[0]]
        py_feats = tsfeatures(y_train_df, freq=freq).set_index('unique_id')

    calculate_features_m4('Hourly', 'data', 100)
    calculate_features_m4('Daily', 'data', 100)
