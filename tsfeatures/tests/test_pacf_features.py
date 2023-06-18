#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tsfeatures import pacf_features


def test_pacf_features_seasonal_short():
    z = np.random.normal(size=15)
    pacf_features(z, freq=7)
