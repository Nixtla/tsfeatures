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

# Column-block size used by `hurst_exponent`'s vectorized rescaled-range loop.
# Each block computes an (end, block) matrix, so peak memory stays O(n * block)
# regardless of series length: three float64 arrays of shape (end, block) are
# live at once (the raw difference `v` plus the two `np.where` masked copies
# used for the max/min reductions), i.e. peak bytes ~= 3 * n * block * 8.
# Measured RSS deltas: ~2.9MB at n=1e3, ~33MB at n=1e4, ~326MB at n=1e5 (block
# stays fixed at 128, so this grows linearly in n, O(n * block), never the
# O(n**2) a single full-broadcast matrix would need). This eliminates the
# per-step Python-level call overhead that dominates for short-to-medium
# series. Block size was chosen empirically as a good trade-off between
# per-block numpy overhead (favors bigger blocks) and wasted "future" rows
# computed at the start of each block (favors smaller blocks); it gives >1x
# speedup at every series length we benchmarked (200 to ~10,000 points),
# never a regression versus the pure Python loop.
_HURST_BLOCK = 128


def hurst_exponent(x: np.array) -> float:
    """Computes hurst exponent.

    Parameters
    ----------
    x: numpy array
        Time series.

    Notes
    -----
    The internal rescaled-range computation centers the input once
    (`x - x.mean()`) before taking any cumulative sums. Both the `s_t`
    (prefix standard deviation) and `r_t` (prefix range) terms are
    algebraically shift-invariant, so centering does not change the result
    versus computing on the raw series. What it does change is numerical
    conditioning: `s_t` is computed via the `E[X**2] - E[X]**2` identity,
    which catastrophically cancels when the *raw* series has a large mean
    relative to its variance (e.g. mean ~1e6, std ~1e-3), producing negative
    pre-clip values and ultimately NaN. Centering first means the identity's
    precision now depends only on the within-series dynamic range (how much
    the series varies internally), not on any constant offset applied to
    the whole series -- the well-conditioned case for this identity,
    regardless of how large the raw values happen to be.

    References
    ----------
    [1] Taken from https://gist.github.com/alexvorndran/aad69fa741e579aad093608ccaab4fe1
    [2] Based on https://codereview.stackexchange.com/questions/224360/hurst-exponent-calculator
    """
    n = x.size  # num timesteps
    t = np.arange(1, n + 1, dtype=np.float64)
    # Center once, up front: every prefix identity below (s_t via the
    # E[X**2]-E[X]**2 shortcut, and r_t's range term) is invariant to a
    # constant shift of the whole series, so this doesn't change the result,
    # it only keeps the cumulative sums well-conditioned regardless of the
    # raw series' magnitude (see the docstring "Notes" section above).
    x0 = x - x.mean()
    y = np.cumsum(x0, dtype=np.float64)  # marginally more efficient than: np.cumsum(sig)
    mean_t = y / t  # running mean (of the centered series)

    # s_t[i] = sqrt(mean_{k<=i} (x0[k] - mean_t[i])**2)
    #        = sqrt(mean_{k<=i}(x0[k]**2) - mean_t[i]**2)
    # The second form is the standard E[X**2] - E[X]**2 identity, computed in
    # O(n) with a single cumulative sum of x0**2 instead of an O(n) Python
    # loop of O(n) numpy calls (one `mean()` per prefix). It is not
    # bit-identical to the two-pass formula above, but the two are equal up
    # to floating point error. Because x0 is already centered, the identity's
    # cancellation is governed by the series' internal dynamic range rather
    # than any additive offset in the raw data.
    mean_x2_t = np.cumsum(x0 * x0, dtype=np.float64) / t
    s_t = np.sqrt(np.clip(mean_x2_t - mean_t * mean_t, 0.0, None))

    # r_t[i] = ptp_{k<=i} (y[k] - t[k] * mean_t[i])
    # No closed form is used here: the term being ranged depends on both the
    # prefix length i (through mean_t[i]) and the row k, so it can't be
    # reduced to a running max/min the way s_t can. Instead we batch prefixes
    # into column blocks and evaluate each block with numpy broadcasting
    # instead of one call per prefix; this is the same arithmetic (same
    # subtractions and max/min reductions) just computed in bulk, so results
    # are numerically identical to the direct per-prefix loop (max/min are
    # order-independent, unlike a sum).
    r_t = np.empty(n, dtype=np.float64)

    for start in range(0, n, _HURST_BLOCK):
        end = min(start + _HURST_BLOCK, n)
        cols = np.arange(start, end)               # prefix indices i in this block
        rows = np.arange(end)                       # all indices k needed by the block
        valid = rows[:, None] <= cols[None, :]       # (end, end-start), True where k <= i

        mean_t_block = mean_t[start:end]
        v = y[:end, None] - t[:end, None] * mean_t_block[None, :]
        v_max = np.where(valid, v, -np.inf).max(axis=0)
        v_min = np.where(valid, v, np.inf).min(axis=0)
        r_t[start:end] = v_max - v_min

    with np.errstate(invalid='ignore'):
        r_s = r_t / s_t

    r_s = np.log(r_s)[1:]
    log_t = np.log(t)[1:]
    a = np.column_stack((log_t, np.ones(log_t.size)))
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
