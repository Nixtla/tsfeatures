#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from tsfeatures import sparsity, tsfeatures

def test_non_zero_sparsity():
    # if we scale the data, the sparsity should be zero
    z = np.zeros(10)
    z[-1] = 1
    df = pd.DataFrame({'unique_id': 1, 'ds': range(1, 11), 'y': z})
    features = tsfeatures(df, freq=7, scale=True, features=[sparsity])
    z_sparsity = features['sparsity'].values[0]
    assert z_sparsity == 0.


def test_sparsity():
    z = np.zeros(10)
    z[-1] = 1
    df = pd.DataFrame({'unique_id': 1, 'ds': range(1, 11), 'y': z})
    features = tsfeatures(df, freq=7, scale=False, features=[sparsity])
    print(features)
    z_sparsity = features['sparsity'].values[0]
    assert z_sparsity == 0.9
