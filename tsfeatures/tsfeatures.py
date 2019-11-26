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

def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack(list((x**k for k in range(p+1)))))
    return np.linalg.qr(X)[0][:,1:]

def acf_features(x):
    m = 7#x.index.freq
    if m is None:
        m = 1
    nlags_ = max(m, 10)
    acfx = acf(x, nlags = nlags_, fft=False)
    acfdiff1x = acf(np.diff(x, n = 1), nlags =  nlags_, fft=False)
    acfdiff2x = acf(np.diff(x, n = 2), nlags =  nlags_, fft=False)
    # first autocorrelation coefficient
    acf_1 = acfx[1]
    
    # sum of squares of first 10 autocorrelation coefficients
    sum_of_sq_acf10 = np.sum((acfx[1:11])**2)
    
    # first autocorrelation ciefficient of differenced series
    diff1_acf1 = acfdiff1x[1]
    
    # sum of squared of first 10 autocorrelation coefficients of differenced series
    diff1_acf10 = np.sum((acfdiff1x[1:11])**2)
    
    # first autocorrelation coefficient of twice-differenced series
    diff2_acf1 = acfdiff2x[1]
    
    # Sum of squared of first 10 autocorrelation coefficients of twice-differenced series
    diff2_acf10 = np.sum((acfdiff2x[1:11])**2)
    
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
    m = 7#x.index.freq
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
    fit = ExponentialSmoothing(x, trend = 'add').fit()
    params = {
        'alpha': fit.params['smoothing_level'],
        'beta': fit.params['smoothing_slope']
    }
    
    return params


def hw_parameters(x):
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
    try:
        # Maybe 100 can change
        entropy = spectral_entropy(x, 1)
    except:
        entropy = None
        
    return {'entropy': entropy}

def lumpiness(x):
    width = 7 # This must be changed
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
    width = 7 # This must be changed
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

# Time series features based of sliding windows
#def max_level_shift(x):
#    width = 7 # This must be changed
    
    
    
def frequency(x):
    return {'frequency': 7}#x.index.freq}

def scalets(x):
    scaledx = scale(x, axis=0, with_mean=True, with_std=True, copy=True)
    #ts = pd.Series(scaledx, index=x.index)
    return scaledx

def stl_features(x):
    """
    Returns a DF where each column is an statistic.
    """
    ### 1
    # Size of ts
    nperiods = len(x)
    # STL fits
    stlfit = STL(x, period=13).fit()
    trend0 = stlfit.trend
    remainder = stlfit.resid
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
    acfremainder = acf_features(remainder)
    
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


# Main functions
def _get_feats(ts_, features):
    c_map = ChainMap(*[dict_feat for dict_feat in [func(ts_) for func in features]])

    return pd.DataFrame(dict(c_map), index = [0])

def tsfeatures(
            tslist,
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
                  stability
            ],
            scale = True,
            parallel = False,
            threads = None
    ):
    """
    tslist: list of numpy arrays or pandas Series class 
    """
    # Setting initial var for parallel tasks
    if parallel and threads is None:
        threads = mp.cpu_count()
            
    # Scaling
    if scale:
        # Parallel 
        if parallel:
            with mp.Pool(threads) as pool:
                tslist = pool.map(scalets, tslist)
        else:
            tslist = [scalets(ts) for ts in tslist]
        
    
    # Init parallel
    if parallel:
        n_series = len(tslist)
        with mp.Pool(mp.cpu_count()) as pool: 
            ts_features = pool.starmap(_get_feats, zip(tslist, [features for i in range(n_series)]))
    else:
        ts_features = [_get_feats(ts, features) for ts in tslist]
    
    
    feat_df = pd.concat(ts_features).reset_index(drop=True)
    
    return feat_df