import numpy as np
import pytest

from tsfeatures import heterogeneity


@pytest.mark.parametrize("freq", [1, 12])
@pytest.mark.parametrize("autoregressive", [False, True])
def test_heterogeneity_supported_statsmodels(freq, autoregressive):
    x = np.random.RandomState(42).normal(size=500)
    if autoregressive:
        for i in range(2, len(x)):
            x[i] += 0.6 * x[i - 1] - 0.3 * x[i - 2]
    original = x.copy()

    result = heterogeneity(x, freq=freq)

    assert set(result) == {"arch_acf", "garch_acf", "arch_r2", "garch_r2"}
    assert all(np.isfinite(value) for value in result.values())
    np.testing.assert_array_equal(x, original)
