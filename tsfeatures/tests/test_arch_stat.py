from tsfeatures import arch_stat
from tsfeatures.utils_ts import WWWusage, USAccDeaths
from math import isclose

def test_arch_stat_seasonal():
    z = arch_stat((USAccDeaths, 12))
    assert isclose(len(z), 1)
    assert isclose(z['arch_lm'], 0.54, abs_tol=0.01)

def test_arch_stat_non_seasonal():
    z = arch_stat((WWWusage, 12))
    assert isclose(len(z), 1)
    assert isclose(z['arch_lm'], 0.98, abs_tol=0.01)
