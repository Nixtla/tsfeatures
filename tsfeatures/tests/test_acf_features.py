#!/usr/bin/env python
# coding: utf-8

from math import isclose
from tsfeatures import acf_features
from tsfeatures.utils import WWWusage, USAccDeaths

def test_acf_features_seasonal():
    z = acf_features(USAccDeaths, 12)
    assert isclose(len(z), 7)
    assert isclose(z['x_acf1'], 0.70, abs_tol=0.01)
    assert isclose(z['x_acf10'], 1.20, abs_tol=0.01)
    assert isclose(z['diff1_acf1'], 0.023, abs_tol=0.01)
    assert isclose(z['diff1_acf10'], 0.27, abs_tol=0.01)
    assert isclose(z['diff2_acf1'], -0.48, abs_tol=0.01)
    assert isclose(z['diff2_acf10'], 0.74, abs_tol=0.01)
    assert isclose(z['seas_acf1'], 0.62, abs_tol=0.01)

def test_acf_features_non_seasonal():
    z = acf_features(WWWusage, 1)
    assert isclose(len(z), 6)
    assert isclose(z['x_acf1'], 0.96, abs_tol=0.01)
    assert isclose(z['x_acf10'], 4.19, abs_tol=0.01)
    assert isclose(z['diff1_acf1'], 0.79, abs_tol=0.01)
    assert isclose(z['diff1_acf10'], 1.40, abs_tol=0.01)
    assert isclose(z['diff2_acf1'], 0.17, abs_tol=0.01)
    assert isclose(z['diff2_acf10'], 0.33, abs_tol=0.01)
