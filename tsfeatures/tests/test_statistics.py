#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from tsfeatures import statistics, tsfeatures

def test_scale():
    z = np.zeros(10)
    z[-1] = 1
    df = pd.DataFrame({'unique_id': 1, 'ds': range(1, 11), 'y': z})
    features = tsfeatures(df, freq=7, scale=True, features=[statistics])
    print(features)

def test_no_scale():
    z = np.zeros(10)
    z[-1] = 1
    df = pd.DataFrame({'unique_id': 1, 'ds': range(1, 11), 'y': z})
    features = tsfeatures(df, freq=7, scale=False, features=[statistics])
    print(features)


if __name__=="__main__":
    test_scale()
    test_no_scale()
