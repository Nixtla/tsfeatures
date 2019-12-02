#from tsfeatures.funcs import *
from sklearn.preprocessing import scale
import pandas as pd
from collections import ChainMap
from statsmodels.tsa.seasonal import STL
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from entropy import spectral_entropy
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from itertools import groupby
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf
from arch import arch_model
import logging

def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack(list((x**k for k in range(p+1)))))
    return np.linalg.qr(X)[0][:,1:]

def acf_features(x):
    ### Unpacking series
    (x, m) = x
    if m is None:
        m = 1
    size_x = len(x)
    
    acfx = acf(x, nlags = max(size_x, 10), fft=False)
    if size_x > 10:
        acfdiff1x = acf(np.diff(x, n = 1), nlags =  10, fft=False)
    else:
        acfdiff1x = np.nan
        
    if size_x > 11:
        acfdiff2x = acf(np.diff(x, n = 2), nlags =  10, fft=False)
    else:
        acfdiff2x = np.nan    
    
    # first autocorrelation coefficient
    acf_1 = acfx[1]
    
    # sum of squares of first 10 autocorrelation coefficients
    sum_of_sq_acf10 = np.sum((acfx[:11])**2)
    
    # first autocorrelation ciefficient of differenced series
    diff1_acf1 = acfdiff1x[1]
    
    # sum of squared of first 10 autocorrelation coefficients of differenced series
    diff1_acf10 = np.sum((acfdiff1x[:11])**2)
    
    # first autocorrelation coefficient of twice-differenced series
    diff2_acf1 = acfdiff2x[1]
    
    # Sum of squared of first 10 autocorrelation coefficients of twice-differenced series
    diff2_acf10 = np.sum((acfdiff2x[:11])**2)
    
    output = {
        'x_acf1': acf_1,
        'x_acf10': sum_of_sq_acf10,
        'diff1_acf1': diff1_acf1,
        'diff1_acf10': diff1_acf10,
        'diff2_acf1': diff2_acf1,
        'diff2_acf10': diff2_acf10
    }
    
    if m > 1:
        output['seas_acf1'] = acfx[m + 2]
    
    return output

def pacf_features(x):
    """
    Partial autocorrelation function features.
    """
    ### Unpacking series
    (x, m) = x
    if m is None:
        m = 1
    nlags_ = max(m, 5)
    
    pacfx = acf(x, nlags = nlags_, fft=False)
    
    # Sum of first 6 PACs squared
    if len(x) > 5:
        pacf_5 = np.sum(pacfx[:4]**2)
    else:
        pacf_5 = None
    
    # Sum of first 5 PACs of difference series squared
    if len(x) > 6:
        diff1_pacf_5 = np.sum(pacf(np.diff(x, n = 1), nlags = 5)**2)
    else:
        diff1_pacf_5 = None
        
        
    # Sum of first 5 PACs of twice differenced series squared
    if len(x) > 7:
        diff2_pacf_5 = np.sum(pacf(np.diff(x, n = 1), nlags = 5)**2)
    else:
        diff2_pacf_5 = None
    
    output = {
        'x_pacf5': pacf_5,
        'diff1x_pacf5': diff1_pacf_5,
        'diff2x_pacf5': diff1_pacf_5
    }
    
    if m > 1:
        output['seas_pacf'] = pacfx[m]
    
    return output

def holt_parameters(x):
    ### Unpacking series
    (x, m) = x
    fit = ExponentialSmoothing(x, trend = 'add').fit()
    params = {
        'alpha': fit.params['smoothing_level'],
        'beta': fit.params['smoothing_slope']
    }
    
    return params


def hw_parameters(x):
    ### Unpacking series
    (x, m) = x
    # Hack: ExponentialSmothing needs a date index
    # this must be fixed
    dates_hack = pd.date_range(end = '2019-01-01', periods = len(x))
    fit = ExponentialSmoothing(x, trend = 'add', seasonal = 'add', dates = dates_hack).fit()
    params = {
        'hwalpha': fit.params['smoothing_level'],
        'hwbeta': fit.params['smoothing_slope'],
        'hwgamma': fit.params['smoothing_seasonal']
    }
    
    return params

# features

def entropy(x):
    ### Unpacking series
    (x, m) = x
    try:
        # Maybe 100 can change
        entropy = spectral_entropy(x, 1)
    except:
        entropy = np.nan
        
    return {'entropy': entropy}

def lumpiness(x):
    ### Unpacking series
    (x, width) = x
    nr = len(x)
    lo = np.arange(1, nr, width)
    up = np.arange(width, nr + width, width)
    nsegs = nr / width
    #print(np.arange(nsegs))
    varx = [np.var(x[lo[idx]:up[idx]]) for idx in np.arange(int(nsegs))]
    
    if len(x) < 2*width:
        lumpiness = 0
    else:
        lumpiness = np.var(varx)
        
    return {'lumpiness': lumpiness}

def stability(x):
    ### Unpacking series
    (x, width) = x
    nr = len(x)
    lo = np.arange(1, nr, width)
    up = np.arange(width, nr + width, width)
    nsegs = nr / width
    #print(np.arange(nsegs))
    meanx = [np.mean(x[lo[idx]:up[idx]]) for idx in np.arange(int(nsegs))]
    
    if len(x) < 2*width:
        stability = 0
    else:
        stability = np.var(meanx)
        
    return {'stability': stability}

def crossing_points(x):
    (x, m) = x
    midline = np.median(x)
    ab = x <= midline
    lenx = len(x)
    p1 = ab[:(lenx-1)]
    p2 = ab[1:]
    cross = (p1 & (~p2)) | (p2 & (~p1))
    return {'crossing_points': cross.sum()}

def flat_spots(x):
    (x, m) = x
    try:
        cutx = pd.cut(x, bins=10, include_lowest=True, labels=False) + 1
    except:
        return {'flat_spots': np.nan}

    rlex = np.array([sum(1 for i in g) for k,g in groupby(cutx)]).max()
    
    return {'flat_spots': rlex}

def heterogeneity(x):
    (x, m) = x
    size_x = len(x)
    order_ar = min(size_x-1, 10*np.log10(size_x)).astype(int) # Defaults for
    x_whitened = AR(x).fit(maxlag = order_ar).resid

    # arch and box test
    x_archtest = arch_stat((x_whitened, m))['arch_lm']
    LBstat = (acf(x_whitened**2, nlags=12, fft=False)[1:]**2).sum()

    #Fit garch model
    garch_fit = arch_model(x_whitened, vol='GARCH', rescale=False).fit(disp='off')

    # compare arch test before and after fitting garch
    garch_fit_std = garch_fit.resid
    x_garch_archtest = arch_stat((garch_fit_std, m))['arch_lm']

    # compare Box test of squared residuals before and after fittig.garch
    LBstat2 = (acf(garch_fit_std**2, nlags=12, fft=False)[1:]**2).sum()
    
    output = {
        'arch_acf': LBstat,
        'garch_acf': LBstat2,
        'arch_2': x_archtest,
        'garch_r2': x_garch_archtest
    }
     
    return output

def series_length(x):
    (x, m) = x
    
    return {'series_length': len(x)}
# Time series features based of sliding windows
#def max_level_shift(x):
#    width = 7 # This must be changed
    

def frequency(x):
    ### Unpacking series
    (x, m) = x
    # Needs frequency of series
    return {'frequency': m}#x.index.freq}

def scalets(x):
    # Scaling time series
    scaledx = scale(x, axis=0, with_mean=True, with_std=True, copy=True)
    #ts = pd.Series(scaledx, index=x.index)
    return scaledx

def stl_features(x):
    """
    Returns a DF where each column is an statistic.
    """
    ### Unpacking series
    (x, m) = x
    # Size of ts
    nperiods = m > 1
    # STL fits
    #print(x)
    stlfit = STL(x, period=m).fit()
    trend0 = stlfit.trend
    remainder = stlfit.resid
    #print(len(remainder))
    seasonal = stlfit.seasonal
    
    # De-trended and de-seasonalized data
    detrend = x - trend0
    deseason = x - seasonal
    fits = x - remainder
    
    # Summay stats
    n = len(x)
    varx = np.nanvar(x)
    vare = np.nanvar(remainder)
    vardetrend = np.nanvar(detrend)
    vardeseason = np.nanvar(deseason)
    
    #Measure of trend strength
    if varx < np.finfo(float).eps:
        trend = 0
    elif (vardeseason/varx < 1e-10):
        trend = 0
    else:
        trend = max(0, min(1, 1 - vare/vardeseason))

    # Measure of seasonal strength
    if varx < np.finfo(float).eps:
        seasonality = 0
    elif np.nanvar(remainder + seasonal) < np.finfo(float).eps:
        seasonality = 0
    else:
        seasonality = max(0, min(1, 1 - vare/np.nanvar(remainder + seasonal)))
  

    # Compute measure of spikiness
    d = (remainder - np.nanmean(remainder))**2
    varloo = (vare*(n-1)-d)/(n-2)
    spike = np.nanvar(varloo)
    
    # Compute measures of linearity and curvature 
    time = np.arange(n) + 1
    poly_m = poly(time, 2)
    time_x = sm.add_constant(poly_m)
    coefs = sm.OLS(trend0, time_x).fit().params
    
    linearity = coefs[1]
    curvature = coefs[2]
    
    # ACF features
    acfremainder = acf_features((remainder, m))
    
    # Assemble features
    output = {
        'nperiods': nperiods,
        'seasonal_period': 1,
        'trend': trend,
        'spike': spike,
        'linearity': linearity,
        'curvature': curvature,
        'e_acf1': acfremainder['x_acf1'],
        'e_acf10': acfremainder['x_acf10']   
    }
    
    return output

#### Heterogeneity coefficients

#ARCH LM statistic
def arch_stat(x, lags=12, demean=True):
    (x, m) = x
    if len(x) <= 13:
        return {'arch_lm': np.nan}
    if demean:
        x = x - np.mean(x)
    
    size_x = len(x)
    slice_ = size_x - lags
    xx = x**2
    y = xx[:slice_]
    X = np.roll(xx, -lags)[:slice_].reshape(-1, 1)
    
    try:
        r_squared = LinearRegression().fit(X, y).score(X.reshape(-1, 1), y)
    except:
        r_squared = np.nan
    
    return {'arch_lm': r_squared}

# Main functions
def _get_feats(tuple_ts_features):
    (ts_, features) = tuple_ts_features
    c_map = ChainMap(*[dict_feat for dict_feat in [func(ts_) for func in features]])

    return pd.DataFrame(dict(c_map), index = [0])

def tsfeatures(
            tslist,
            frcy,
            features = [
                stl_features, 
                frequency, 
                entropy, 
                acf_features,
                pacf_features,
                #holt_parameters,
                #hw_parameters,
                entropy, 
                lumpiness,
                stability,
                arch_stat,
                series_length,
                heterogeneity,
                flat_spots,
                crossing_points
            ],
            scale = True,
            parallel = False,
            threads = None
    ):
    """
    tslist: list of numpy arrays or pandas Series class 
    """
    if not isinstance(tslist, list):
        tslist = [tslist]

            
    # Scaling
    if scale:
        # Parallel 
        if parallel:
            with mp.Pool(threads) as pool:
                tslist = pool.map(scalets, tslist)
        else:
            tslist = [scalets(ts) for ts in tslist]
        
    
    # There's methods which needs frequency
    # This is a hack for this
    # each feature function receives a tuple (ts, frcy)
    tslist = [(ts, frcy) for ts in tslist]
    
    # Init parallel
    if parallel:
        n_series = len(tslist)
        with mp.Pool(threads) as pool: 
            ts_features = pool.map(_get_feats, zip(tslist, [features for i in range(n_series)]))
    else:
        ts_features = [_get_feats((ts, features)) for ts in tslist]
    
    
    feat_df = pd.concat(ts_features).reset_index(drop=True)
    
    return feat_df
