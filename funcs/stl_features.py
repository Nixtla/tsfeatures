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