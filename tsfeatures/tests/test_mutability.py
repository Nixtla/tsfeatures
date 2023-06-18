import numpy as np
import pandas as pd
from tsfeatures import (
    tsfeatures, acf_features, arch_stat, crossing_points,
    entropy, flat_spots, heterogeneity, holt_parameters,
    lumpiness, nonlinearity, pacf_features, stl_features,
    stability, hw_parameters, unitroot_kpss, unitroot_pp,
    series_length, sparsity, hurst
)


def test_mutability():
    z = np.zeros(100)
    z[-1] = 1
    z_df = pd.DataFrame({'unique_id': 1, 'ds': range(1, 101), 'y': z})
    feats=[sparsity, acf_features, arch_stat, crossing_points,
              entropy, flat_spots, holt_parameters,
              lumpiness, nonlinearity, pacf_features, stl_features,
              stability, hw_parameters, unitroot_kpss, unitroot_pp,
              series_length, hurst]
    feats_2=[acf_features, arch_stat, crossing_points,
              entropy, flat_spots, holt_parameters,
              lumpiness, nonlinearity, pacf_features, stl_features,
              stability, hw_parameters, unitroot_kpss, unitroot_pp,
              series_length, hurst, sparsity]
    feats_df = tsfeatures(z_df, freq=7, features=feats, scale=False)
    feats_2_df = tsfeatures(z_df, freq=7, features=feats_2, scale=False)
    pd.testing.assert_frame_equal(feats_df, feats_2_df[feats_df.columns])

if __name__=="__main__":
    test_mutability()
