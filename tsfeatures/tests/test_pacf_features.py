#!/usr/bin/env python
# coding: utf-8

from math import isclose
from tsfeatures import pacf_features
from tsfeatures.utils import WWWusage, USAccDeaths

def test_pacf_features_seasonal():
    z = pacf_features(USAccDeaths, 12)
    assert isclose(len(z), 4)
    assert isclose(z['x_pacf5'], 0.63, abs_tol=0.01)
    assert isclose(z['diff1x_pacf5'], 0.09, abs_tol=0.01)
    assert isclose(z['diff2x_pacf5'], 0.38, abs_tol=0.01)
    assert isclose(z['seas_pacf'], 0.12, abs_tol=0.01)

def test_pacf_features_non_seasonal():
    z = pacf_features(WWWusage, 1)
    assert isclose(len(z), 3)
    assert isclose(z['x_pacf5'], 1.03, abs_tol=0.01)
    assert isclose(z['diff1x_pacf5'], 0.80, abs_tol=0.01)
    assert isclose(z['diff2x_pacf5'], 0.22, abs_tol=0.01)
