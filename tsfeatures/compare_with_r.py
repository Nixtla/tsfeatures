#!/usr/bin/env python
# coding: utf-8

import argparse
import time
import cProfile
import pstats
import sys

from tsfeatures import tsfeatures
from tsfeatures.tsfeatures_r import tsfeatures_r
from tsfeatures.m4_data import prepare_m4_data

FREQS = {'Hourly': 24, 'Daily': 1,
         'Monthly': 12, 'Quarterly': 4,
         'Weekly':1, 'Yearly': 1}


def compare_features_m4(dataset_name, directory, num_obs=1000000):
    _, y_train_df, _, _ = prepare_m4_data(dataset_name=dataset_name,
                                          directory = directory,
                                          num_obs=num_obs)

    freq = FREQS[dataset_name]

    print('Calculating python features...')

    init = time.time()
    py_feats = tsfeatures(y_train_df, freq=freq)

    print('Total time: ', time.time() - init)

    print('Calculating r features...')
    init = time.time()
    r_feats = tsfeatures_r(y_train_df, freq=freq)

    print('Total time: ', time.time() - init)

    diff = (py_feats - r_feats).abs().sum(0).sort_values()

    return diff

def main(args):
    if args.num_obs:
        num_obs = args.num_obs
    else:
        num_obs = 100000

    if args.dataset_name:
        datasets = [args.dataset_name]
    else:
        datasets = FREQS.keys()

    for dataset_name in datasets:
        diff = compare_features_m4(dataset_name, args.results_directory, num_obs)
        print(diff)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Get features for M4 data')
    parser.add_argument("--results_directory", required=True, type=str,
                        help="directory where M4 data will be downloaded")
    parser.add_argument("--num_obs", required=False, type=int,
                        help="number of M4 time series to be tested (uses all data by default)")
    parser.add_argument("--dataset_name", required=False, type=str,
                        help="type of dataset to get features")
    args = parser.parse_args()

    pr = cProfile.Profile()
    pr.enable()
    main(args)

    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('time')
    stats.print_stats(0)
