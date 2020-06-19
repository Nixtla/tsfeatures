#!/usr/bin/env python
# coding: utf-8

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
except ImportError:
    raise ImportError('rpy2 package not found, please install it.')

def tsfeatures_r(ts, freq, **kwargs):
    """tsfeatures wrapper using r.

    Parameters
    ----------
    ts: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
    freq: int
        Frequency of the time series.
    **kwargs:
        Arguments passed to the original tsfeatures function.

    References
    ----------
    https://pkg.robjhyndman.com/tsfeatures/reference/tsfeatures.html
    """
    rstring = """
        function(df, freq, ...){
          suppressMessages(library(data.table))
          suppressMessages(library(tsfeatures))

          set.seed(12398)

          dt <- as.data.table(df)
          setkey(dt, unique_id)

          series_list <- split(dt, by = "unique_id", keep.by = FALSE)
          series_list <- lapply(series_list,
                                function(serie) serie[, ts(y, frequency = freq)])

          features <-
            c("length",
              "acf_features",
              "arch_stat",
              "crossing_points",
              "entropy",
              "flat_spots",
              "heterogeneity",
              "holt_parameters",
              "hurst",
              "hw_parameters",
              "lumpiness",
              "nonlinearity",
              "pacf_features",
              "stability",
              "stl_features",
              "unitroot_kpss",
              "unitroot_pp")

          series_features <- tsfeatures(series_list)#, features = features)

          setDT(series_features)

          series_features[, unique_id := names(series_list)]
        }
    """

    pandas2ri.activate()
    rfunc = robjects.r(rstring)

    feats = rfunc(ts, freq, **kwargs)

    feats = feats.set_index('unique_id')

    return feats
