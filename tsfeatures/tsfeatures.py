
import warnings
warnings.warn = lambda *a, **kw: False

import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from collections import ChainMap
from rstl import STL

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, kpss
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import Holt

from entropy import spectral_entropy
import multiprocessing as mp

from sklearn.linear_model import LinearRegression
from itertools import groupby
from arch import arch_model
#from supersmoother import SuperSmoother
from supsmu import supsmu
from functools import partial

from tsfeatures.utils_ts import poly, embed, scalets
from tsfeatures.custom_tests import terasvirta_test, sample_entropy, \
                                    hurst_exponent, ur_pp


def acf_features(x, freq=None):
    ### Unpacking series
    if freq is None:
        m = 1
    else:
        m = freq
    size_x = len(x)

    acfx = acf(x, nlags = max(m, 10), fft=False)
    if size_x > 10:
        acfdiff1x = acf(np.diff(x, n = 1), nlags =  10, fft=False)
    else:
        acfdiff1x = [np.nan]*2

    if size_x > 11:
        acfdiff2x = acf(np.diff(x, n = 2), nlags =  10, fft=False)
    else:
        acfdiff2x = [np.nan]*2

    # first autocorrelation coefficient
    acf_1 = acfx[1]

    # sum of squares of first 10 autocorrelation coefficients
    sum_of_sq_acf10 = np.sum((acfx[1:11])**2) if size_x > 10 else np.nan

    # first autocorrelation ciefficient of differenced series
    diff1_acf1 = acfdiff1x[1]

    # sum of squared of first 10 autocorrelation coefficients of differenced series
    diff1_acf10 = np.sum((acfdiff1x[1:11])**2) if size_x > 10 else np.nan

    # first autocorrelation coefficient of twice-differenced series
    diff2_acf1 = acfdiff2x[1]

    # Sum of squared of first 10 autocorrelation coefficients of twice-differenced series
    diff2_acf10 = np.sum((acfdiff2x[1:11])**2) if size_x > 11 else np.nan

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

def pacf_features(x, freq=None):
    """
    Partial autocorrelation function features.
    """
    if freq is None:
        m = 1
    else:
        m = freq

    nlags_ = max(m, 5)

    if len(x) > 1:
        try:
            pacfx = pacf(x, nlags = nlags_, method='ldb')
        except:
            pacfx = np.nan
    else:
        pacfx = np.nan

    # Sum of first 6 PACs squared
    if len(x) > 5:
        pacf_5 = np.sum(pacfx[1:6]**2)
    else:
        pacf_5 = np.nan

    # Sum of first 5 PACs of difference series squared
    if len(x) > 6:
        try:
            diff1_pacf = pacf(np.diff(x, n = 1), nlags = 5, method='ldb')[1:6]
            diff1_pacf_5 = np.sum(diff1_pacf**2)
        except:
            diff1_pacf_5 = np.nan
    else:
        diff1_pacf_5 = np.nan


    # Sum of first 5 PACs of twice differenced series squared
    if len(x) > 7:
        try:
            diff2_pacf = pacf(np.diff(x, n = 2), nlags = 5, method='ldb')[1:6]
            diff2_pacf_5 = np.sum(diff2_pacf**2)
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

def holt_parameters(x, freq=None):
    try :
        fit = Holt(x).fit()
        params = {
            'alpha': fit.params['smoothing_level'],
            'beta': fit.params['smoothing_slope']
        }
    except:
        params = {
            'alpha': np.nan,
            'beta': np.nan
        }

    return params


def hw_parameters(x, freq=None):
    if freq is None:
        m = 1
    else:
        m = freq
    try:
        fit = ExponentialSmoothing(x, seasonal_periods=m, trend='add', seasonal='add').fit()
        params = {
            'hw_alpha': fit.params['smoothing_level'],
            'hw_beta': fit.params['smoothing_slope'],
            'hw_gamma': fit.params['smoothing_seasonal']
        }
    except:
        params = {
            'hw_alpha': np.nan,
            'hw_beta': np.nan,
            'hw_gamma': np.nan
        }
    return params

# features

def entropy(x, freq=None):
    try:
        # Maybe 100 can change
        entropy = spectral_entropy(x, freq, method='welch', normalize=True)
    except:
        entropy = np.nan

    return {'entropy': entropy}

def count_entropy(x, freq=None):
    entropy = x[x>0]*np.log(x[x>0])
    entropy = -entropy.sum()

    return {'entropy': entropy}

def lumpiness(x, freq=None):

    if (freq == 1) or (freq is None):
        width = 10
    else:
        width = freq

    nr = len(x)
    lo = np.arange(0, nr, width)
    up = lo + width
    nsegs = nr / width
    varx = [np.nanvar(x[lo[idx]:up[idx]], ddof=1) for idx in np.arange(int(nsegs))]

    if len(x) < 2*width:
        lumpiness = 0
    else:
        lumpiness = np.nanvar(varx, ddof=1)

    return {'lumpiness': lumpiness}

def stability(x, freq=None):
    if freq == 1:
        width = 10
    else:
        width = freq

    nr = len(x)
    lo = np.arange(0, nr, width)
    up = lo + width
    nsegs = nr / width
    #print(np.arange(nsegs))
    meanx = [np.nanmean(x[lo[idx]:up[idx]]) for idx in np.arange(int(nsegs))]

    if len(x) < 2*width:
        stability = 0
    else:
        stability = np.nanvar(meanx, ddof=1)

    return {'stability': stability}

def crossing_points(x, freq=None):
    midline = np.median(x)
    ab = x <= midline
    lenx = len(x)
    p1 = ab[:(lenx-1)]
    p2 = ab[1:]
    cross = (p1 & (~p2)) | (p2 & (~p1))
    return {'crossing_points': cross.sum()}

def flat_spots(x, freq=None):
    try:
        cutx = pd.cut(x, bins=10, include_lowest=True, labels=False) + 1
    except:
        return {'flat_spots': np.nan}

    rlex = np.array([sum(1 for i in g) for k,g in groupby(cutx)]).max()

    return {'flat_spots': rlex}

def heterogeneity(x, freq=None):

    if freq is None:
        m = 1
    else:
        m = freq

    size_x = len(x)
    order_ar = min(size_x-1, np.floor(10*np.log10(size_x))).astype(int) # Defaults for
    try:
        x_whitened = AR(x).fit(maxlag = order_ar, ic = 'aic', trend='c').resid
    except:
        x_whitened = AR(x).fit(maxlag = order_ar, ic = 'aic', trend='nc').resid

    # arch and box test
    x_archtest = arch_stat(x_whitened, m)['arch_lm']
    LBstat = (acf(x_whitened**2, nlags=12, fft=False)[1:]**2).sum()

    #Fit garch model
    garch_fit = arch_model(x_whitened, vol='GARCH', rescale=False).fit(disp='off')

    # compare arch test before and after fitting garch
    garch_fit_std = garch_fit.resid
    x_garch_archtest = arch_stat(garch_fit_std, m)['arch_lm']

    # compare Box test of squared residuals before and after fittig.garch
    LBstat2 = (acf(garch_fit_std**2, nlags=12, fft=False)[1:]**2).sum()

    output = {
        'arch_acf': LBstat,
        'garch_acf': LBstat2,
        'arch_r2': x_archtest,
        'garch_r2': x_garch_archtest
    }

    return output

def series_length(x, freq=None):

    return {'series_length': len(x)}


# Unit root test statistics
def unitroot_kpss(x, freq=None):
    n = len(x)
    nlags = int(4 * (n / 100)**(1 / 4))

    try:
        test_kpss, _, _, _ = kpss(x, nlags=nlags)
    except:
        test_kpss = np.nan

    return {'unitroot_kpss': test_kpss}

def unitroot_pp(x, freq=None):
    try:
        test_pp = ur_pp(x)
    except:
        test_pp = np.nan

    return {'unitroot_pp': test_pp}


def nonlinearity(x, freq=None):
    try:
        test = terasvirta_test(x)
        test = 10*test/len(x)
    except:
        test = np.nan

    return {'nonlinearity': test}


def frequency(x, freq=None):
    if freq is None:
        m = 1
    else:
        m = freq
    # Needs frequency of series
    return {'frequency': m}#x.index.freq}

def stl_features(x, freq=None):
    """
    Returns a DF where each column is an statistic.
    """
    if freq is None:
        m = 1
    else:
        m = freq
    nperiods = int(m > 1)
    # STL fits
    if m > 1:
        try:
            stlfit = STL(np.array(x), m, 13)
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

        trend0 = stlfit.trend
        remainder = stlfit.remainder
        seasonal = stlfit.seasonal
    else:
        deseas = np.array(x)
        t = np.arange(len(x))+1
        try:
            trend0 = supsmu(t, deseas)
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
            season = max(0, min(1, 1 - vare/np.nanvar(remainder + seasonal, ddof=1)))

        peak = (np.argmax(seasonal)+1) % m
        peak = m if peak == 0 else peak

        trough = (np.argmin(seasonal)+1) % m
        trough = m if trough == 0 else trough



    # Compute measure of spikiness
    d = (remainder - np.nanmean(remainder))**2
    varloo = (vare*(n-1)-d)/(n-2)
    spike = np.nanvar(varloo, ddof=1)

    # Compute measures of linearity and curvature
    time = np.arange(n) + 1
    poly_m = poly(time, 2)
    time_x = sm.add_constant(poly_m)
    coefs = sm.OLS(trend0, time_x).fit().params

    linearity = coefs[1]
    curvature = -coefs[2] #if m > 1 else coefs[2]

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

    if m>1:
        output['seasonal_strength'] = season
        output['peak'] = peak
        output['trough'] = trough

    return output

def sparsity(x, freq=None):
    return {'sparsity': np.mean(x == 0)}

#### Heterogeneity coefficients

#ARCH LM statistic
def arch_stat(x, freq=None, lags=12, demean=True):
    if len(x) <= lags+1:
        return {'arch_lm': np.nan}
    if demean:
        x -= np.mean(x)

    size_x = len(x)
    mat = embed(x**2, lags+1)
    X = mat[:,1:]
    y = np.vstack(mat[:, 0])

    try:
        r_squared = LinearRegression().fit(X, y).score(X, y)
    except:
        r_squared = np.nan

    return {'arch_lm': r_squared}

def series_length(x, freq=None):
    return {'series_length': len(x)}

def hurst(x, freq=None):

    try:
        hurst_index = hurst_exponent(x)
    except:
        hurst_index = np.nan

    return {'hurst': hurst_index}

# Main functions
def get_feats(index, ts, freq, scale=True,
              features = [acf_features, arch_stat, crossing_points,
                          entropy, flat_spots, heterogeneity, holt_parameters,
                          lumpiness, nonlinearity, pacf_features, stl_features,
                          stability, hw_parameters, unitroot_kpss, unitroot_pp,
                          series_length, hurst]):


    if isinstance(ts, pd.DataFrame):
        assert 'y' in ts.columns
        ts = ts['y'].values

    if isinstance(ts, pd.Series):
        ts = ts.values

    if scale:
        ts = scalets(ts)

    c_map = ChainMap(*[dict_feat for dict_feat in [func(ts, freq) for func in features]])

    return pd.DataFrame(dict(c_map), index = [index])

def tsfeatures(
            ts,
            freq,
            features = [acf_features, arch_stat, crossing_points,
                        entropy, flat_spots, heterogeneity, holt_parameters,
                        lumpiness, nonlinearity, pacf_features, stl_features,
                        stability, hw_parameters, unitroot_kpss, unitroot_pp,
                        series_length, hurst],
            scale = True,
            parallel = False,
            threads = None
    ):
    """
    ts: pandas df
            panel with columns unique_id, ds, y
    """

    partial_get_feats = partial(get_feats, freq=freq, scale=scale, features=features)
    # Init parallel
    if parallel:
        n_series = len(ts)

        with mp.Pool(threads) as pool:
            ts_features = pool.starmap(partial_get_feats, ts.groupby('unique_id'))
    else:
        ts_features = [partial_get_feats(idx, df) for idx, df in ts.groupby('unique_id')]


    feat_df = pd.concat(ts_features)

    return feat_df
