#from tsfeatures.funcs import *
from sklearn.preprocessing import scale
import pandas as pd
from collections import ChainMap
from stldecompose import decompose, forecast
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack((x**k for k in range(p+1))))
    return np.linalg.qr(X)[0][:,1:]

def acf_features(x):
    m = x.index.freq
    if m is None:
        m = 1
    nlags_ = max(m, 10)
    acfx = acf(x, nlags = nlags_)
    acfdiff1x = acf(np.diff(x, n = 1), nlags =  nlags_)
    acfdiff2x = acf(np.diff(x, n = 2), nlags =  nlags_)
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


def entropy(x):
    try:
        # Maybe 100 can change
        entropy = spectral_entropy(x, 1)
    except:
        entropy = None
        
    return {'entropy': entropy}

def frequency(x):
    return {'frequency': x.index.freq}

def scalets(x):
    scaledx = scale(x, axis=0, with_mean=True, with_std=True, copy=True)
    ts = pd.Series(scaledx, index=x.index)
    return ts

def stl_features(x):
    """
    Returns a DF where each column is an statistic.
    """
    ### 1
    # Size of ts
    nperiods = len(x)
    # STL fits
    stlfit = decompose(x, period=13)
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

def tsfeatures(tslist,
              features = [stl_features, frequency, entropy, acf_features],
              scale = True):
    if scale:
        tslist = [scalets(ts) for ts in tslist]
    
    feat_df = pd.concat(
        [
            pd.DataFrame(
                dict(
                    ChainMap(*[dict_feat for dict_feat in [func(ts) for func in features]])
                ),
                index = [0]
            ) for ts in tslist
        ]
    ).reset_index(drop=True)
    
    return feat_df