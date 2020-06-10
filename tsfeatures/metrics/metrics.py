#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import multiprocessing as mp

from math import sqrt
from functools import partial
from dask import delayed, compute

AVAILABLE_METRICS = ['mse', 'rmse', 'mape', 'smape', 'mase', 'rmsse']

######################################################################
# METRICS
######################################################################

def mse(y, y_hat):
    """Calculates Mean Squared Error.

    MSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array
        predicted values

    Return
    ------
    scalar: MSE
    """
    mse = np.mean(np.square(y - y_hat))
    return mse

def rmse(y, y_hat):
    """Calculates Root Mean Squared Error.

    RMSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: RMSE
    """
    rmse = sqrt(np.mean(np.square(y - y_hat)))
    return rmse

def mape(y, y_hat):
    """Calculates Mean Absolute Percentage Error.

    MAPE measures the relative prediction accuracy of a
    forecasting method by calculating the percentual deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: MAPE
    """
    mape = np.mean(np.abs(y - y_hat) / np.abs(y))
    mape = 100 * mape
    return mape

def smape(y, y_hat):
    """Calculates Symmetric Mean Absolute Percentage Error.

    SMAPE measures the relative prediction accuracy of a
    forecasting method by calculating the relative deviation
    of the prediction and the true value scaled by the sum of the
    absolute values for the prediction and true value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: SMAPE
    """
    smape = np.mean(np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))
    smape = 200 * smape
    return smape

def mase(y, y_hat, y_train, seasonality=1):
    """Calculates the M4 Mean Absolute Scaled Error.

    MASE measures the relative prediction accuracy of a
    forecasting method by comparinng the mean absolute errors
    of the prediction and the true value against the mean
    absolute errors of the seasonal naive model.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values
    y_train: numpy array
      actual train values for Naive1 predictions
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1

    Return
    ------
    scalar: MASE
    """
    scale = np.mean(abs(y_train[seasonality:] - y_train[:-seasonality]))
    mase = np.mean(abs(y - y_hat)) / scale
    mase = 100 * mase
    return mase


def rmsse(y, y_hat, y_train, seasonality=1):
    """Calculates the M5 Root Mean Squared Scaled Error.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1

    Return
    ------
    scalar: RMSSE
    """
    scale = np.mean(np.square(y_train[seasonality:] - y_train[:-seasonality]))
    rmsse = sqrt(mse(y, y_hat) / scale)
    rmsse = 100 * rmsse
    return rmsse

def mini_owa(y, y_hat, y_train, seasonality, y_bench):
    """Calculates the Overall Weighted Average for a single series.

    MASE, sMAPE for Naive2 and current model
    then calculatess Overall Weighted Average.

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array of len h (forecasting horizon)
        predicted values
    seasonality: int
        main frequency of the time series
        Hourly 24,  Daily 7, Weekly 52,
        Monthly 12, Quarterly 4, Yearly 1
    y_train: numpy array
        insample values of the series for scale
    y_bench: numpy array of len h (forecasting horizon)
        predicted values of the benchmark model

    Return
    ------
    return: mini_OWA
    """
    mase_y = mase(y, y_hat, y_train, seasonality)
    mase_bench = mase(y, y_bench, y_train, seasonality)

    smape_y = smape(y, y_hat)
    smape_bench = smape(y, y_bench)

    mini_owa = ((mase_y/mase_bench) + (smape_y/smape_bench))/2

    return mini_owa

######################################################################
# PANEL EVALUATION
######################################################################

def _evaluate_ts(uid, y_test, y_hat,
                 y_train, metric,
                 seasonality, y_bench, metric_name):
    y_test_uid = y_test.loc[uid].y.values
    y_hat_uid = y_hat.loc[uid].y_hat.values

    if metric_name in ['mase', 'rmsse']:
        y_train_uid = y_train.loc[uid].y.values
        evaluation_uid = metric(y=y_test_uid, y_hat=y_hat_uid,
                                y_train=y_train_uid,
                                seasonality=seasonality)
    elif metric_name in ['mini_owa']:
        y_train_uid = y_train.loc[uid].y.values
        y_bench_uid = y_bench.loc[uid].y_hat.values
        evaluation_uid = metric(y=y_test_uid, y_hat=y_hat_uid,
                                y_train=y_train_uid,
                                seasonality=seasonality,
                                y_bench=y_bench_uid)

    else:
         evaluation_uid = metric(y=y_test_uid, y_hat=y_hat_uid)

    return uid, evaluation_uid

def evaluate_panel(y_test, y_hat, y_train,
                   metric, seasonality=None,
                   y_bench=None,
                   threads=None):
    """Calculates a specific metric for y and y_hat (and y_train, if needed).

    Parameters
    ----------
    y_test: pandas df
        df with columns ['unique_id', 'ds', 'y']
    y_hat: pandas df
        df with columns ['unique_id', 'ds', 'y_hat']
    y_train: pandas df
        df with columns ['unique_id', 'ds', 'y'] (train)
        This is used in the scaled metrics ('mase', 'rmsse').
    seasonality: int
        Main frequency of the time series.
        Used in ('mase', 'rmsse').
        Commonly used seasonalities:
            Hourly: 24,
            Daily: 7,
            Weekly: 52,
            Monthly: 12,
            Quarterly: 4,
            Yearly: 1.
    y_bench: pandas df
        df with columns ['unique_id', 'ds', 'y_hat']
        predicted values of the benchmark model
        This is used in 'mini_owa'.
    threads: int
        Number of threads to use. Use None (default) for parallel processing.

    Return
    ------
    list of metric evaluations for each unique_id
        in the panel data
    """
    metric_name = metric.__code__.co_name
    uids = y_test['unique_id'].unique()
    y_hat_uids = y_hat['unique_id'].unique()
    assert len(y_test)==len(y_hat), "not same length"
    assert all(uids == y_hat_uids), "not same u_ids"


    y_test = y_test.set_index(['unique_id', 'ds'])
    y_hat = y_hat.set_index(['unique_id', 'ds'])

    if metric_name in ['mase', 'rmsse']:
        y_train = y_train.set_index(['unique_id', 'ds'])
    elif metric_name in ['mini_owa']:
        y_train = y_train.set_index(['unique_id', 'ds'])
        y_bench = y_bench.set_index(['unique_id', 'ds'])

    partial_evaluation = partial(_evaluate_ts, y_test=y_test, y_hat=y_hat,
                                 y_train=y_train, metric=metric,
                                 seasonality=seasonality,
                                 y_bench=y_bench,
                                 metric_name=metric_name)

    with mp.Pool(threads) as pool:
        evaluations = pool.map(partial_evaluation, uids)

    evaluations = pd.DataFrame(evaluations, columns=['unique_id', 'error'])


    return evaluations
