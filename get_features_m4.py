#!/usr/bin/env python
# coding: utf-8

import argparse

from ESRNN.m4_data import prepare_m4_data
from ESRNN.utils_evaluation import evaluate_prediction_owa
from ESRNN import ESRNN

from tsfeatures import tsfeatures


freqs = {'Hourly': 24, 'Daily': 1,
         'Monthly': 1, 'Quarterly': 4,
         'Weekly':1,'Yearly':1}

def get_features_m4(dataset_name, directory, num_obs=1000000, parallel=True):
    print('\n')
    print(dataset_name)
    _, y_train_df, _, _ = prepare_m4_data(dataset_name=dataset_name,
                                          directory = directory,
                                          num_obs=num_obs)

    freq = freqs[dataset_name]
    feats = tsfeatures(y_train_df, freq=freq, parallel=parallel)
    feats = feats.rename_axis('unique_id')

    feats.to_csv('{}/results/{}-features.csv'.format(directory, dataset_name))

    print('Features saved')

def main(args):
    if args.num_obs:
        num_obs = args.num_obs
    else:
        num_obs = 100000

    if args.dataset_name:
        datasets = [args.dataset_name]
    else:
        datasets = freqs.keys()

    for dataset_name in datasets:
        get_features_m4(dataset_name, args.results_directory, num_obs)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Get features for M4 data')
    parser.add_argument("--results_directory", required=True, type=str,
                        help="directory where M4 data will be downloaded")
    parser.add_argument("--num_obs", required=False, type=int,
                        help="number of M4 time series to be tested (uses all data by default)")
    parser.add_argument("--dataset_name", required=False, type=str,
                        help="type of dataset to get features")
    args = parser.parse_args()

    main(args)
