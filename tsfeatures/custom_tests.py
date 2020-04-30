from tsfeatures.utils_ts import embed, scalets
import numpy as np
import statsmodels.api as sm


def terasvirta_test(x, lag=1, scale=True):
    """
    x: array
    """

    if scale: x = scalets(x)

    size_x = len(x)
    y = embed(x, lag+1)

    X = y[:, 1:]
    X = sm.add_constant(X)

    y = y[:, 0]

    ols = sm.OLS(y, X).fit()

    u = ols.resid
    ssr0 = (u**2).sum()

    X_nn_list = []

    for i in range(lag):
        for j in range(i, lag):
            element = X[:, i+1]*X[:, j+1]
            element = np.vstack(element)
            X_nn_list.append(element)

    for i in range(lag):
        for j in range(i, lag):
            for k in range(j, lag):
                element = X[:, i+1]*X[:, j+1]*X[:, k+1]
                element = np.vstack(element)
                X_nn_list.append(element)


    X_nn = np.concatenate(X_nn_list, axis=1)
    X_nn = np.concatenate([X, X_nn], axis=1)
    ols_nn = sm.OLS(u, X_nn).fit()

    v = ols_nn.resid
    ssr = (v**2).sum()

    stat = size_x*np.log(ssr0/ssr)

    return stat

def sample_entropy(x):
    """
    Calculate and return sample entropy of x.
    .. rubric:: References
    |  [1] http://en.wikipedia.org/wiki/Sample_Entropy
    |  [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.array(x)

    sample_length = 1  # number of sequential points of the time series
    tolerance = 0.2 * np.std(x)  # 0.2 is a common value for r - why?

    n = len(x)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((1, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((1, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = x[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) / 2
    B = np.vstack(([N], B[0]))

    # sample entropy = -1 * (log (A/B))
    similarity_ratio = A / B
    se = -1 * np.log(similarity_ratio)
    se = np.reshape(se, -1)
    return se[0]

def hurst_ernie_chan(p, lags=12):
    #taken from
    #https://stackoverflow.com/questions/39488806/hurst-exponent-in-python

    variancetau = []; tau = []

    for lag in range(2, lags):

        #  Write the different lags into a vector to compute a set of tau or lags
        tau.append(lag)

        # Compute the log returns on all days, then compute the variance on the difference in log returns
        # call this pp or the price difference
        pp = np.subtract(p[lag:], p[:-lag])
        variancetau.append(np.var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.
    #print tau
    #print variancetau

    # plot the log of those variance against the log of tau and get the slope
    m = np.polyfit(np.log10(tau),np.log10(variancetau),1)

    hurst = m[0] / 2

    return hurst

def ur_pp(x):
    n = len(x)
    lmax = 4 * (n / 100)**(1 / 4)

    lmax, _ = divmod(lmax, 1)
    lmax = int(lmax)

    y, y_l1 = x[1:], x[:(n-1)]

    n-=1

    y_l1 = sm.add_constant(y_l1)

    model = sm.OLS(y, y_l1).fit()
    my_tstat, res = model.tvalues[0], model.resid
    s = 1 / (n * np.sum(res**2))
    myybar = (1/n**2)*(((y-y.mean())**2).sum())
    myy = (1/n**2)*((y**2).sum())
    my = (n**(-3/2))*(y.sum())

    idx = np.arange(lmax)
    coprods = []
    for i in idx:
        first_del = res[(i+1):]
        sec_del = res[:(n-i-1)]
        prod = first_del*sec_del
        coprods.append(prod.sum())
    coprods = np.array(coprods)

    weights = 1 - (idx+1)/(lmax+1)
    sig = s + (2/n)*((weights*coprods).sum())
    lambda_ = 0.5*(sig-s)
    lambda_prime = lambda_/sig

    alpha = model.params[1]

    test_stat = n*(alpha-1)-lambda_/myybar

    return test_stat
