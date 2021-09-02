#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.warn = lambda *a, **kw: False
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from collections import ChainMap
from functools import partial
from itertools import groupby
from math import log, e
from multiprocessing import cpu_count, Pool
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd
from antropy import spectral_entropy
from arch import arch_model
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from statsmodels.api import add_constant, OLS
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf, kpss
from supersmoother import SuperSmoother

from .utils import (embed, FREQS, hurst_exponent,
                    lambda_coef_var, poly,
                    scalets, terasvirta_test, ur_pp)


def acf_features(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Calculates autocorrelation function features.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'x_acf1': First autocorrelation coefficient.
        'x_acf10': Sum of squares of first 10 autocorrelation coefficients.
        'diff1_acf1': First autocorrelation ciefficient of differenced series.
        'diff1_acf10': Sum of squared of first 10 autocorrelation coefficients
                       of differenced series.
        'diff2_acf1': First autocorrelation coefficient of twice-differenced series.
        'diff2_acf10': Sum of squared of first 10 autocorrelation coefficients of
                       twice-differenced series.

        Only for seasonal data (freq > 1).
        'seas_acf1': Autocorrelation coefficient at the first seasonal lag.
    """
    m = freq
    size_x = len(x)

    acfx = acf(x, nlags=max(m, 10), fft=False)
    if size_x > 10:
        acfdiff1x = acf(np.diff(x, n=1), nlags=10, fft=False)
    else:
        acfdiff1x = [np.nan]*2

    if size_x > 11:
        acfdiff2x = acf(np.diff(x, n=2), nlags=10, fft=False)
    else:
        acfdiff2x = [np.nan] * 2
    # first autocorrelation coefficient
    acf_1 = acfx[1]
    # sum of squares of first 10 autocorrelation coefficients
    sum_of_sq_acf10 = np.sum((acfx[1:11]) ** 2) if size_x > 10 else np.nan
    # first autocorrelation ciefficient of differenced series
    diff1_acf1 = acfdiff1x[1]
    # sum of squared of first 10 autocorrelation coefficients of differenced series
    diff1_acf10 = np.sum((acfdiff1x[1:11]) ** 2) if size_x > 10 else np.nan
    # first autocorrelation coefficient of twice-differenced series
    diff2_acf1 = acfdiff2x[1]
    # Sum of squared of first 10 autocorrelation coefficients of twice-differenced series
    diff2_acf10 = np.sum((acfdiff2x[1:11]) ** 2) if size_x > 11 else np.nan

    output = {
        'x_acf1': acf_1,
        'x_acf10': sum_of_sq_acf10,
        'diff1_acf1': diff1_acf1,
        'diff1_acf10': diff1_acf10,
        'diff2_acf1': diff2_acf1,
        'diff2_acf10': diff2_acf10
    }

    if m > 1:
        output['seas_acf1'] = acfx[m] if len(acfx) > m else np.nan

    return output

def arch_stat(x: np.array, freq: int = 1,
              lags: int = 12, demean: bool = True) -> Dict[str, float]:
    """Arch model features.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'arch_lm': R^2 value of an autoregressive model of order lags applied to x**2.
    """
    if len(x) <= lags + 1:
        return {'arch_lm': np.nan}
    if demean:
        x -= np.mean(x)

    size_x = len(x)
    mat = embed(x ** 2, lags + 1)
    X = mat[:, 1:]
    y = np.vstack(mat[:, 0])

    try:
        r_squared = LinearRegression().fit(X, y).score(X, y)
    except:
        r_squared = np.nan

    return {'arch_lm': r_squared}

def count_entropy(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Count entropy.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'count_entropy': Entropy using only positive data.
    """
    entropy = x[x > 0] * np.log(x[x > 0])
    entropy = -entropy.sum()

    return {'count_entropy': entropy}

def crossing_points(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Crossing points.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'crossing_points': Number of times that x crosses the median.
    """
    midline = np.median(x)
    ab = x <= midline
    lenx = len(x)
    p1 = ab[:(lenx - 1)]
    p2 = ab[1:]
    cross = (p1 & (~p2)) | (p2 & (~p1))

    return {'crossing_points': cross.sum()}

def entropy(x: np.array, freq: int = 1, base: float = e) -> Dict[str, float]:
    """Calculates sample entropy.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'entropy': Wrapper of the function spectral_entropy.
    """
    try:
        with np.errstate(divide='ignore'):
            entropy = spectral_entropy(x, 1, normalize=True)
    except:
        entropy = np.nan

    return {'entropy': entropy}

def flat_spots(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Flat spots.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'flat_spots': Number of flat spots in x.
    """
    try:
        cutx = pd.cut(x, bins=10, include_lowest=True, labels=False) + 1
    except:
        return {'flat_spots': np.nan}

    rlex = np.array([sum(1 for i in g) for k,g in groupby(cutx)]).max()

    return {'flat_spots': rlex}

def frequency(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Frequency.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'frequency': Wrapper of freq.
    """

    return {'frequency': freq}

def guerrero(x: np.array, freq: int = 1,
             lower: int = -1, upper: int = 2) -> Dict[str, float]:
    """Applies Guerrero's (1993) method to select the lambda which minimises the
    coefficient of variation for subseries of x.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series.
    lower: float
        The lower bound for lambda.
    upper: float
        The upper bound for lambda.

    Returns
    -------
    dict
        'guerrero': Minimum coefficient of variation for subseries of x.

    References
    ----------
    [1] Guerrero, V.M. (1993) Time-series analysis supported by power transformations.
        Journal of Forecasting, 12, 37–48.
    """
    func_to_min = lambda lambda_par: lambda_coef_var(lambda_par, x=x, period=freq)

    min_ = minimize_scalar(func_to_min, bounds=[lower, upper])
    min_ = min_['fun']

    return {'guerrero': min_}

def heterogeneity(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Heterogeneity.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'arch_acf': Sum of squares of the first 12 autocorrelations of the
                    residuals of the AR model applied to x
        'garch_acf': Sum of squares of the first 12 autocorrelations of the
                    residuals of the GARCH model applied to x
        'arch_r2': Function arch_stat applied to the residuals of the
                   AR model applied to x.
        'garch_r2': Function arch_stat applied to the residuals of the GARCH
                    model applied to x.
    """
    m = freq

    size_x = len(x)
    order_ar = min(size_x - 1, np.floor(10 * np.log10(size_x)))
    order_ar = int(order_ar)

    try:
        x_whitened = AR(x).fit(maxlag=order_ar, ic='aic', trend='c').resid
    except:
        try:
            x_whitened = AR(x).fit(maxlag=order_ar, ic='aic', trend='nc').resid
        except:
            output = {
                'arch_acf': np.nan,
                'garch_acf': np.nan,
                'arch_r2': np.nan,
                'garch_r2': np.nan
            }

            return output
    # arch and box test
    x_archtest = arch_stat(x_whitened, m)['arch_lm']
    LBstat = (acf(x_whitened ** 2, nlags=12, fft=False)[1:] ** 2).sum()
    #Fit garch model
    garch_fit = arch_model(x_whitened, vol='GARCH', rescale=False).fit(disp='off')
    # compare arch test before and after fitting garch
    garch_fit_std = garch_fit.resid
    x_garch_archtest = arch_stat(garch_fit_std, m)['arch_lm']
    # compare Box test of squared residuals before and after fittig.garch
    LBstat2 = (acf(garch_fit_std ** 2, nlags=12, fft=False)[1:] ** 2).sum()

    output = {
        'arch_acf': LBstat,
        'garch_acf': LBstat2,
        'arch_r2': x_archtest,
        'garch_r2': x_garch_archtest
    }

    return output

def holt_parameters(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Fitted parameters of a Holt model.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'alpha': Level paramater of the Holt model.
        'beta': Trend parameter of the Hold model.
    """
    try :
        fit = ExponentialSmoothing(x, trend='add', seasonal=None).fit()
        params = {
            'alpha': fit.params['smoothing_level'],
            'beta': fit.params['smoothing_trend']
        }
    except:
        params = {
            'alpha': np.nan,
            'beta': np.nan
        }

    return params

def hurst(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Hurst index.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'hurst': Hurst exponent.
    """
    try:
        hurst_index = hurst_exponent(x)
    except:
        hurst_index = np.nan

    return {'hurst': hurst_index}

def hw_parameters(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Fitted parameters of a Holt-Winters model.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'hw_alpha': Level parameter of the HW model.
        'hw_beta': Trend parameter of the HW model.
        'hw_gamma': Seasonal parameter of the HW model.
    """
    try:
        fit = ExponentialSmoothing(x, seasonal_periods=freq, trend='add', seasonal='add').fit()
        params = {
            'hw_alpha': fit.params['smoothing_level'],
            'hw_beta': fit.params['smoothing_trend'],
            'hw_gamma': fit.params['smoothing_seasonal']
        }
    except:
        params = {
            'hw_alpha': np.nan,
            'hw_beta': np.nan,
            'hw_gamma': np.nan
        }

    return params

def intervals(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Intervals with demand.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'intervals_mean': Mean of intervals with positive values.
        'intervals_sd': SD of intervals with positive values.
    """
    x[x > 0] = 1

    y = [sum(val) for keys, val in groupby(x, key=lambda k: k != 0) if keys != 0]
    y = np.array(y)

    return {'intervals_mean': np.mean(y), 'intervals_sd': np.std(y, ddof=1)}

def lumpiness(x: np.array, freq: int = 1) -> Dict[str, float]:
    """lumpiness.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'lumpiness': Variance of the variances of tiled windows.
    """
    if freq == 1:
        width = 10
    else:
        width = freq

    nr = len(x)
    lo = np.arange(0, nr, width)
    up = lo + width
    nsegs = nr / width
    varx = [np.nanvar(x[lo[idx]:up[idx]], ddof=1) for idx in np.arange(int(nsegs))]

    if len(x) < 2 * width:
        lumpiness = 0
    else:
        lumpiness = np.nanvar(varx, ddof=1)

    return {'lumpiness': lumpiness}

def nonlinearity(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Nonlinearity.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'nonlinearity': 10 t**2/len(x) where t is the statistic used in
                        Terasvirta's test.
    """
    try:
        test = terasvirta_test(x)
        test = 10 * test / len(x)
    except:
        test = np.nan

    return {'nonlinearity': test}

def pacf_features(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Calculates partial autocorrelation function features.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'x_pacf5':  Sum of squares of the first 5 partial autocorrelation
                    coefficients.
        'diff1x_pacf5': Sum of squares of the first 5 partial autocorrelation
                        coefficients of differenced series.
        'diff2x_pacf5': Sum of squares of the first 5 partial autocorrelation
                        coefficients of twice-differenced series.

        Only for seasonal data (freq > 1).
        'seas_pacf': Partial autocorrelation
                     coefficient at the first seasonal lag.
    """
    m = freq

    nlags_ = max(m, 5)

    if len(x) > 1:
        try:
            pacfx = pacf(x, nlags=nlags_, method='ldb')
        except:
            pacfx = np.nan
    else:
        pacfx = np.nan
    # Sum of first 6 PACs squared
    if len(x) > 5:
        pacf_5 = np.sum(pacfx[1:6] ** 2)
    else:
        pacf_5 = np.nan
    # Sum of first 5 PACs of difference series squared
    if len(x) > 6:
        try:
            diff1_pacf = pacf(np.diff(x, n=1), nlags=5, method='ldb')[1:6]
            diff1_pacf_5 = np.sum(diff1_pacf ** 2)
        except:
            diff1_pacf_5 = np.nan
    else:
        diff1_pacf_5 = np.nan
    # Sum of first 5 PACs of twice differenced series squared
    if len(x) > 7:
        try:
            diff2_pacf = pacf(np.diff(x, n = 2), nlags = 5, method='ldb')[1:6]
            diff2_pacf_5 = np.sum(diff2_pacf ** 2)
        except:
            diff2_pacf_5 = np.nan
    else:
        diff2_pacf_5 = np.nan

    output = {
        'x_pacf5': pacf_5,
        'diff1x_pacf5': diff1_pacf_5,
        'diff2x_pacf5': diff2_pacf_5
    }

    if m > 1:
        output['seas_pacf'] = pacfx[m] if len(pacfx) > m else np.nan

    return output

def series_length(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Series length.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'series_length': Wrapper of len(x).
    """

    return {'series_length': len(x)}

def sparsity(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Sparsity.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'sparsity': Average obs with zero values.
    """

    return {'sparsity': np.mean(x == 0)}

def stability(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Stability.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'stability': Variance of the means of tiled windows.
    """
    if freq == 1:
        width = 10
    else:
        width = freq

    nr = len(x)
    lo = np.arange(0, nr, width)
    up = lo + width
    nsegs = nr / width
    meanx = [np.nanmean(x[lo[idx]:up[idx]]) for idx in np.arange(int(nsegs))]

    if len(x) < 2 * width:
        stability = 0
    else:
        stability = np.nanvar(meanx, ddof=1)

    return {'stability': stability}

def stl_features(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Calculates seasonal trend using loess decomposition.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'nperiods': Number of seasonal periods in x.
        'seasonal_period': Frequency of the time series.
        'trend': Strength of trend.
        'spike': Measures "spikiness" of x.
        'linearity': Linearity of x based on the coefficients of an
                     orthogonal quadratic regression.
        'curvature': Curvature of x based on the coefficients of an
                     orthogonal quadratic regression.
        'e_acf1': acfremainder['x_acf1']
        'e_acf10': acfremainder['x_acf10']

        Only for sesonal data (freq > 0).
        'seasonal_strength': Strength of seasonality.
        'peak': Strength of peaks.
        'trough': Strength of trough.
    """
    m = freq
    nperiods = int(m > 1)
    # STL fits
    if m > 1:
        try:
            stlfit = STL(x, m, 13).fit()
        except:
            output = {
                'nperiods': nperiods,
                'seasonal_period': m,
                'trend': np.nan,
                'spike': np.nan,
                'linearity': np.nan,
                'curvature': np.nan,
                'e_acf1': np.nan,
                'e_acf10': np.nan,
                'seasonal_strength': np.nan,
                'peak': np.nan,
                'trough': np.nan
            }

            return output

        trend0 = stlfit.trend
        remainder = stlfit.resid
        seasonal = stlfit.seasonal
    else:
        deseas = x
        t = np.arange(len(x)) + 1
        try:
            trend0 = SuperSmoother().fit(t, deseas).predict(t)
        except:
            output = {
                'nperiods': nperiods,
                'seasonal_period': m,
                'trend': np.nan,
                'spike': np.nan,
                'linearity': np.nan,
                'curvature': np.nan,
                'e_acf1': np.nan,
                'e_acf10': np.nan
            }

            return output

        remainder = deseas - trend0
        seasonal = np.zeros(len(x))
    # De-trended and de-seasonalized data
    detrend = x - trend0
    deseason = x - seasonal
    fits = x - remainder
    # Summay stats
    n = len(x)
    varx = np.nanvar(x, ddof=1)
    vare = np.nanvar(remainder, ddof=1)
    vardetrend = np.nanvar(detrend, ddof=1)
    vardeseason = np.nanvar(deseason, ddof=1)
    #Measure of trend strength
    if varx < np.finfo(float).eps:
        trend = 0
    elif (vardeseason/varx < 1e-10):
        trend = 0
    else:
        trend = max(0, min(1, 1 - vare/vardeseason))
    # Measure of seasonal strength
    if m > 1:
        if varx < np.finfo(float).eps:
            season = 0
        elif np.nanvar(remainder + seasonal, ddof=1) < np.finfo(float).eps:
            season = 0
        else:
            season = max(0, min(1, 1 - vare / np.nanvar(remainder + seasonal, ddof=1)))

        peak = (np.argmax(seasonal) + 1) % m
        peak = m if peak == 0 else peak

        trough = (np.argmin(seasonal) + 1) % m
        trough = m if trough == 0 else trough
    # Compute measure of spikiness
    d = (remainder - np.nanmean(remainder)) ** 2
    varloo = (vare * (n-1) - d) / (n - 2)
    spike = np.nanvar(varloo, ddof=1)
    # Compute measures of linearity and curvature
    time = np.arange(n) + 1
    poly_m = poly(time, 2)
    time_x = add_constant(poly_m)
    coefs = OLS(trend0, time_x).fit().params

    linearity = coefs[1]
    curvature = -coefs[2]
    # ACF features
    acfremainder = acf_features(remainder, m)
    # Assemble features
    output = {
        'nperiods': nperiods,
        'seasonal_period': m,
        'trend': trend,
        'spike': spike,
        'linearity': linearity,
        'curvature': curvature,
        'e_acf1': acfremainder['x_acf1'],
        'e_acf10': acfremainder['x_acf10']
    }

    if m > 1:
        output['seasonal_strength'] = season
        output['peak'] = peak
        output['trough'] = trough

    return output

def unitroot_kpss(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Unit root kpss.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'unitroot_kpss': Statistic for the Kwiatowski et al unit root test.
    """
    n = len(x)
    nlags = int(4 * (n / 100) ** (1 / 4))

    try:
        test_kpss, _, _, _ = kpss(x, nlags=nlags)
    except:
        test_kpss = np.nan

    return {'unitroot_kpss': test_kpss}

def unitroot_pp(x: np.array, freq: int = 1) -> Dict[str, float]:
    """Unit root pp.

    Parameters
    ----------
    x: numpy array
        The time series.
    freq: int
        Frequency of the time series

    Returns
    -------
    dict
        'unitroot_pp': Statistic for the Phillips-Perron unit root test.
    """
    try:
        test_pp = ur_pp(x)
    except:
        test_pp = np.nan

    return {'unitroot_pp': test_pp}

###############################################################################
#### MAIN FUNCTIONS ###########################################################
###############################################################################

def _get_feats(index,
               ts,
               freq,
               scale = True,
               features = [acf_features, arch_stat, crossing_points,
                          entropy, flat_spots, heterogeneity, holt_parameters,
                          lumpiness, nonlinearity, pacf_features, stl_features,
                          stability, hw_parameters, unitroot_kpss, unitroot_pp,
                          series_length, hurst],
                dict_freqs = FREQS):

    if freq is None:
        inf_freq = pd.infer_freq(ts['ds'])
        if inf_freq is None:
            raise Exception(
                'Failed to infer frequency from the `ds` column, '
                'please provide the frequency using the `freq` argument.'
            )

        freq = dict_freqs.get(inf_freq)
        if freq is None:
            raise Exception(
                'Error trying to convert infered frequency from the `ds` column '
                'to integer. Please provide a dictionary with that frequency '
                'as key and the integer frequency as value. '
                f'Infered frequency: {inf_freq}'
            )


    if isinstance(ts, pd.DataFrame):
        assert 'y' in ts.columns
        ts = ts['y'].values

    if isinstance(ts, pd.Series):
        ts = ts.values

    if scale:
        ts = scalets(ts)

    c_map = ChainMap(*[dict_feat for dict_feat in [func(ts, freq) for func in features]])

    return pd.DataFrame(dict(c_map), index = [index])

def tsfeatures(ts: pd.DataFrame,
               freq: Optional[int] = None,
               features: List[Callable] = [acf_features, arch_stat, crossing_points,
                                           entropy, flat_spots, heterogeneity,
                                           holt_parameters, lumpiness, nonlinearity,
                                           pacf_features, stl_features, stability,
                                           hw_parameters, unitroot_kpss, unitroot_pp,
                                           series_length, hurst],
               dict_freqs: Dict[str, int] = FREQS,
               scale: bool = True,
               threads: Optional[int] = None) -> pd.DataFrame:
    """Calculates features for time series.

    Parameters
    ----------
    ts: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y'].
        Long panel of time series.
    freq: int
        Frequency of the time series. If None the frequency of
        each time series is infered and assigns the seasonal periods according to
        dict_freqs.
    features: iterable
        Iterable of features functions.
    scale: bool
        Whether (mean-std)scale data.
    dict_freqs: dict
        Dictionary that maps string frequency of int. Ex: {'D': 7, 'W': 1}
    threads: int
        Number of threads to use. Use None (default) for parallel processing.

    Returns
    -------
    pandas df
        Pandas DataFrame where each column is a feature and each row
        a time series.
    """
    partial_get_feats = partial(_get_feats, freq=freq, scale=scale,
                                features=features, dict_freqs=dict_freqs)

    with Pool(threads) as pool:
        ts_features = pool.starmap(partial_get_feats, ts.groupby('unique_id'))

    ts_features = pd.concat(ts_features).rename_axis('unique_id')
    ts_features = ts_features.reset_index()

    return ts_features

################################################################################
#### MAIN WIDE FUNCTION ########################################################
################################################################################

def _get_feats_wide(index,
                    ts,
                    scale = True,
                    features = [acf_features, arch_stat, crossing_points,
                                entropy, flat_spots, heterogeneity, holt_parameters,
                                lumpiness, nonlinearity, pacf_features, stl_features,
                                stability, hw_parameters, unitroot_kpss, unitroot_pp,
                                series_length, hurst]):
    seasonality = ts['seasonality'].item()
    y = ts['y'].item()
    y = np.array(y)

    if scale:
        y = scalets(y)

    c_map = ChainMap(*[dict_feat for dict_feat in [func(y, seasonality) for func in features]])

    return pd.DataFrame(dict(c_map), index = [index])

def tsfeatures_wide(ts: pd.DataFrame,
                    features: List[Callable] = [acf_features, arch_stat, crossing_points,
                                                entropy, flat_spots, heterogeneity,
                                                holt_parameters, lumpiness, nonlinearity,
                                                pacf_features, stl_features, stability,
                                                hw_parameters, unitroot_kpss, unitroot_pp,
                                                series_length, hurst],
                    scale: bool = True,
                    threads: Optional[int] = None) -> pd.DataFrame:
    """Calculates features for time series.

    Parameters
    ----------
    ts: pandas df
        Pandas DataFrame with columns ['unique_id', 'seasonality', 'y'].
        Wide panel of time series.
    features: iterable
        Iterable of features functions.
    scale: bool
        Whether (mean-std)scale data.
    threads: int
        Number of threads to use. Use None (default) for parallel processing.

    Returns
    -------
    pandas df
        Pandas DataFrame where each column is a feature and each row
        a time series.
    """
    partial_get_feats = partial(_get_feats_wide, scale=scale,
                                features=features)

    with Pool(threads) as pool:
        ts_features = pool.starmap(partial_get_feats, ts.groupby('unique_id'))

    ts_features = pd.concat(ts_features).rename_axis('unique_id')
    ts_features = ts_features.reset_index()

    return ts_features
