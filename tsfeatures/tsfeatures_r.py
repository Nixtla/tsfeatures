#!/usr/bin/env python
# coding: utf-8

from typing import List

import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

def tsfeatures_r(ts: pd.DataFrame,
                 freq: int,
                 features: List[str] = ["length", "acf_features", "arch_stat",
                                        "crossing_points", "entropy", "flat_spots",
                                        "heterogeneity", "holt_parameters",
                                        "hurst", "hw_parameters", "lumpiness",
                                        "nonlinearity", "pacf_features", "stability",
                                        "stl_features", "unitroot_kpss", "unitroot_pp"],
                 **kwargs) -> pd.DataFrame:
    """tsfeatures wrapper using r.

    Parameters
    ----------
    ts: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y'].
        Long panel of time series.
    freq: int
        Frequency of the time series.
    features: List[str]
        String list of features to calculate.
    **kwargs:
        Arguments used by the original tsfeatures function.

    References
    ----------
    https://pkg.robjhyndman.com/tsfeatures/reference/tsfeatures.html
    """
    rstring = """
        function(df, freq, features, ...){
          suppressMessages(library(data.table))
          suppressMessages(library(tsfeatures))

          dt <- as.data.table(df)
          setkey(dt, unique_id)

          series_list <- split(dt, by = "unique_id", keep.by = FALSE)
          series_list <- lapply(series_list,
                                function(serie) serie[, ts(y, frequency = freq)])

          if("hw_parameters" %in% features){
            features <- setdiff(features, "hw_parameters")

            if(length(features)>0){
                hw_series_features <- suppressMessages(tsfeatures(series_list, "hw_parameters", ...))
                names(hw_series_features) <- paste0("hw_", names(hw_series_features))

                series_features <- suppressMessages(tsfeatures(series_list, features, ...))
                series_features <- cbind(series_features, hw_series_features)
            } else {
                series_features <- suppressMessages(tsfeatures(series_list, "hw_parameters", ...))
                names(series_features) <- paste0("hw_", names(series_features))
            }
          } else {
            series_features <- suppressMessages(tsfeatures(series_list, features, ...))
          }

          setDT(series_features)

          series_features[, unique_id := names(series_list)]

        }
    """
    pandas2ri.activate()
    rfunc = robjects.r(rstring)

    feats = rfunc(ts, freq, features, **kwargs)
    pandas2ri.deactivate()

    renamer={'ARCH.LM': 'arch_lm', 'length': 'series_length'}
    feats = feats.rename(columns=renamer)

    return feats

def tsfeatures_r_wide(ts: pd.DataFrame,
                      features: List[str] = ["length", "acf_features", "arch_stat",
                                             "crossing_points", "entropy", "flat_spots",
                                             "heterogeneity", "holt_parameters",
                                             "hurst", "hw_parameters", "lumpiness",
                                             "nonlinearity", "pacf_features", "stability",
                                             "stl_features", "unitroot_kpss", "unitroot_pp"],
                      **kwargs) -> pd.DataFrame:
    """tsfeatures wrapper using r.

    Parameters
    ----------
    ts: pandas df
        Pandas DataFrame with columns ['unique_id', 'seasonality', 'y'].
        Wide panel of time series.
    features: List[str]
        String list of features to calculate.
    **kwargs:
        Arguments used by the original tsfeatures function.

    References
    ----------
    https://pkg.robjhyndman.com/tsfeatures/reference/tsfeatures.html
    """
    rstring = """
        function(uids, seasonalities, ys, features, ...){
            suppressMessages(library(data.table))
            suppressMessages(library(tsfeatures))
            suppressMessages(library(purrr))

            series_list <- pmap(
                                list(uids, seasonalities, ys),
                                function(uid, seasonality, y) ts(y, frequency=seasonality)
                            )
            names(series_list) <- uids

            if("hw_parameters" %in% features){
                features <- setdiff(features, "hw_parameters")

                if(length(features)>0){
                    hw_series_features <- suppressMessages(tsfeatures(series_list, "hw_parameters", ...))
                    names(hw_series_features) <- paste0("hw_", names(hw_series_features))

                    series_features <- suppressMessages(tsfeatures(series_list, features, ...))
                    series_features <- cbind(series_features, hw_series_features)
                } else {
                    series_features <- suppressMessages(tsfeatures(series_list, "hw_parameters", ...))
                    names(series_features) <- paste0("hw_", names(series_features))
                }
            } else {
                series_features <- suppressMessages(tsfeatures(series_list, features, ...))
            }

            setDT(series_features)

            series_features[, unique_id := names(series_list)]

        }
    """
    pandas2ri.activate()
    rfunc = robjects.r(rstring)

    uids = ts['unique_id'].to_list()
    seasonalities = ts['seasonality'].to_list()
    ys = ts['y'].to_list()

    feats = rfunc(uids, seasonalities, ys, features, **kwargs)
    pandas2ri.deactivate()

    renamer={'ARCH.LM': 'arch_lm', 'length': 'series_length'}
    feats = feats.rename(columns=renamer)

    return feats
