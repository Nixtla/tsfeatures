import numpy as np
import pandas as pd
from tsfeatures import (
    tsfeatures, acf_features, arch_stat, crossing_points,
    entropy, flat_spots, heterogeneity, holt_parameters,
    lumpiness, nonlinearity, pacf_features, stl_features,
    stability, hw_parameters, unitroot_kpss, unitroot_pp,
    series_length, sparsity, hurst, statistics
)


def test_small():
    z = np.zeros(2)
    z[-1] = 1
    z_df = pd.DataFrame({'unique_id': 1, 'ds': range(1, 3), 'y': z})
    feats=[sparsity, acf_features, arch_stat, crossing_points,
              entropy, flat_spots, holt_parameters,
              lumpiness, nonlinearity, pacf_features, stl_features,
              stability, hw_parameters, unitroot_kpss, unitroot_pp,
              series_length, hurst, statistics]
    feats_df = tsfeatures(z_df, freq=12, features=feats, scale=False)

def test_small_1():
    z = np.zeros(1)
    z[-1] = 1
    z_df = pd.DataFrame({'unique_id': 1, 'ds': range(1, 2), 'y': z})
    feats=[sparsity, acf_features, arch_stat, crossing_points,
              entropy, flat_spots, holt_parameters,
              lumpiness, nonlinearity, pacf_features, stl_features,
              stability, hw_parameters, unitroot_kpss, unitroot_pp,
              series_length, hurst, statistics]
    feats_df = tsfeatures(z_df, freq=12, features=feats, scale=False)

if __name__=="__main__":
    test_small()
    test_small_1()
