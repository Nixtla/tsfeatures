#!/usr/bin/env python
# coding: utf-8

import numpy as np
import statsmodels.api as sm

from scipy.signal import periodogram, welch

np.seterr(divide='ignore', invalid='ignore')

################################################################################
########### GENERAL UTILS ######################################################
################################################################################

FREQS = {'H': 24, 'D': 1,
         'M': 12, 'Q': 4,
         'W': 1, 'Y': 1}

def scalets(x: np.array) -> float:
    """Mean-std scale."""
    scaledx = (x - x.mean()) / x.std(ddof=1)

    return scaledx

def poly(x: np.array, p: int) -> np.array:
    """Returns or evaluates orthogonal polynomials of degree 1 to degree over the
       specified set of points x:
       these are all orthogonal to the constant polynomial of degree 0.

    Parameters
    ----------
    x: numpy array
        Time series.
    p: int
        Degree of the polynomial.

    References
    ----------
    https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/poly
    """
    X = np.transpose(np.vstack(list((x ** k for k in range(p + 1)))))

    return np.linalg.qr(X)[0][:, 1:]

def embed(x: np.array, p: int) -> np.array:
    """Embeds the time series x into a low-dimensional Euclidean space.

    Parameters
    ----------
    x: numpy array
        Time series.
    p: int
        Embedding dimension.

    References
    ----------
    https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/embed
    """
    x = np.transpose(np.vstack(list((np.roll(x, k) for k in range(p)))))
    x = x[p - 1:]

    return x

################################################################################
####### CUSTOM FUNCS ###########################################################
################################################################################

def terasvirta_test(x: np.array, lag: int = 1, scale: bool = True) -> float:
    """Generically computes Teraesvirta's neural network test for neglected
       nonlinearity either for the time series x or the regression y~x.

    Parameters
    ----------
    x: numpy array
        Time series.
    lag: int
        Specifies the model order in terms of lags.
    scale: bool
        Whether the data should be scaled before computing the test.

    Returns
    -------
    float
        Terasvirta statistic.

    References
    ----------
    https://www.rdocumentation.org/packages/tseries/versions/0.10-47/topics/terasvirta.test
    """
    if scale: x = scalets(x)

    size_x = len(x)
    y = embed(x, lag + 1)

    X = y[:, 1:]
    X = sm.add_constant(X)

    y = y[:, 0]

    ols = sm.OLS(y, X).fit()

    u = ols.resid
    ssr0 = (u ** 2).sum()

    X_nn_list = []

    for i in range(lag):
        for j in range(i, lag):
            element = X[:, i + 1] * X[:, j + 1]
            element = np.vstack(element)
            X_nn_list.append(element)

    for i in range(lag):
        for j in range(i, lag):
            for k in range(j, lag):
                element = X[:, i + 1] * X[:, j + 1] * X[:, k + 1]
                element = np.vstack(element)
                X_nn_list.append(element)

    X_nn = np.concatenate(X_nn_list, axis=1)
    X_nn = np.concatenate([X, X_nn], axis=1)
    ols_nn = sm.OLS(u, X_nn).fit()

    v = ols_nn.resid
    ssr = (v ** 2).sum()

    stat = size_x * np.log(ssr0 / ssr)

    return stat

def hurst_exponent(x: np.array) -> float:
    """Computes hurst exponent.

    Parameters
    ----------
    x: numpy array
        Time series.

    References
    ----------
    [1] Taken from https://gist.github.com/alexvorndran/aad69fa741e579aad093608ccaab4fe1
    [2] Based on https://codereview.stackexchange.com/questions/224360/hurst-exponent-calculator
    """
    n = x.size  # num timesteps
    t = np.arange(1, n + 1)
    y = x.cumsum()  # marginally more efficient than: np.cumsum(sig)
    mean_t = y / t  # running mean

    s_t = np.sqrt(
        np.array([np.mean((x[:i + 1] - mean_t[i]) ** 2) for i in range(n)])
    )
    r_t = np.array([np.ptp(y[:i + 1] - t[:i + 1] * mean_t[i]) for i in range(n)])

    with np.errstate(invalid='ignore'):
        r_s = r_t / s_t

    r_s = np.log(r_s)[1:]
    n = np.log(t)[1:]
    a = np.column_stack((n, np.ones(n.size)))
    hurst_exponent, _ = np.linalg.lstsq(a, r_s, rcond=-1)[0]

    return hurst_exponent

def ur_pp(x: np.array) -> float:
    """Performs the Phillips and Perron unit root test.

    Parameters
    ----------
    x: numpy array
        Time series.

    References
    ----------
    https://www.rdocumentation.org/packages/urca/versions/1.3-0/topics/ur.pp
    """
    n = len(x)
    lmax = 4 * (n / 100) ** (1 / 4)

    lmax, _ = divmod(lmax, 1)
    lmax = int(lmax)

    y, y_l1 = x[1:], x[:n - 1]

    n -= 1

    y_l1 = sm.add_constant(y_l1)

    model = sm.OLS(y, y_l1).fit()
    my_tstat, res = model.tvalues[0], model.resid
    s = 1 / (n * np.sum(res ** 2))
    myybar = (1 / n ** 2) * (((y - y.mean()) ** 2).sum())
    myy = (1 / n ** 2) * ((y ** 2).sum())
    my = (n ** (-3 / 2)) * (y.sum())

    idx = np.arange(lmax)
    coprods = []
    for i in idx:
        first_del = res[i + 1:]
        sec_del = res[:n - i - 1]
        prod = first_del * sec_del
        coprods.append(prod.sum())
    coprods = np.array(coprods)

    weights = 1 - (idx + 1) / (lmax + 1)
    sig = s + (2 / n) * ((weights * coprods).sum())
    lambda_ = 0.5 * (sig - s)
    lambda_prime = lambda_ / sig

    alpha = model.params[1]

    test_stat = n * (alpha - 1) - lambda_ / myybar

    return test_stat

def lambda_coef_var(lambda_par: float, x: np.array, period: int = 2):
    """Calculates coefficient of variation for subseries of x.

    Parameters
    ----------
    lambda_par: float
        Lambda Box-cox transformation parameter.
        Must be greater than zero.
    x: numpy array
        Time series.
    period: int
        The length of each subseries (usually the length of seasonal period).

    Returns
    -------
    float
        Coefficient of variation.
    """
    if len(np.unique(x)) == 1:
        return 1

    split_size = divmod(len(x) - 1, period)
    split_size, _ = split_size

    split = np.array_split(x, split_size)

    mu_h = np.array([np.nanmean(sub) for sub in split])
    sig_h = np.array([np.nanstd(sub, ddof=1) for sub in split])

    rat = sig_h / mu_h ** (1 - lambda_par)

    value = np.nanstd(rat, ddof=1) / np.nanmean(rat)

    return value

################################################################################
####### TS #####################################################################
################################################################################

WWWusage = [88,84,85,85,84,85,83,85,88,89,91,99,104,112,126,
            138,146,151,150,148,147,149,143,132,131,139,147,150,
            148,145,140,134,131,131,129,126,126,132,137,140,142,150,159,
            167,170,171,172,172,174,175,172,172,174,174,169,165,156,142,
            131,121,112,104,102,99,99,95,88,84,84,87,89,88,85,86,89,91,
            91,94,101,110,121,135,145,149,156,165,171,175,177,
            182,193,204,208,210,215,222,228,226,222,220]

USAccDeaths = [9007,8106,8928,9137,10017,10826,11317,10744,9713,9938,9161,
               8927,7750,6981,8038,8422,8714,9512,10120,9823,8743,9129,8710,
               8680,8162,7306,8124,7870,9387,9556,10093,9620,8285,8466,8160,
               8034,7717,7461,7767,7925,8623,8945,10078,9179,8037,8488,7874,
               8647,7792,6957,7726,8106,8890,9299,10625,9302,8314,
               8850,8265,8796,7836,6892,7791,8192,9115,9434,10484,
               9827,9110,9070,8633,9240]
