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