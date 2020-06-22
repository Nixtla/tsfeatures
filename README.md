[![Build](https://github.com/FedericoGarza/tsfeatures/workflows/Python%20package/badge.svg)](https://github.com/FedericoGarza/tsfeatures/tree/master)
[![PyPI version fury.io](https://badge.fury.io/py/tsfeatures.svg)](https://pypi.python.org/pypi/tsfeatures/)
<!-- [![Downloads](https://pepy.tech/badge/tsfeatures)](https://pepy.tech/project/tsfeatures) -->
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360+/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/FedericoGarza/tsfeatures/blob/master/LICENSE)

# tsfeatures

Calculates various features from time series data. Python implementation of the R package _[tsfeatures](https://github.com/robjhyndman/tsfeatures)_.

# Installation

You can install the *released* version of `tsfeatures` from the [Python package index](pypi.org) with:

``` python
pip install tsfeatures
```

# Usage

The `tsfeatures` main function calculates by default the features used by Hyndman et.al. in their implementation of the FFORMA model.

``` python
from tsfeatures import tsfeatures
```

This function receives a panel pandas df with columns `unique_id`, `ds`, `y` and the frequency of the data.

<img src=https://raw.githubusercontent.com/FedericoGarza/tsfeatures/master/.github/images/y_train.png width="152">

``` python
tsfeatures(panel, freq=7)
```

## List of available features

| Features |||
|:--------|:------|:-------------|
|acf_features|heterogeneity|series_length|
|arch_stat|holt_parameters|sparsity|
|count_entropy|hurst|stability|
|crossing_points|hw_parameters|stl_features|
|entropy|intervals|unitroot_kpss|
|flat_spots|lumpiness|unitroot_pp|
|frequency|nonlinearity||
|guerrero|pacf_features||


## Comparison with the R implementation


### Non-seasonal data (100 Daily M4 time series)

| feature         |   diff | feature         |   diff | feature         |   diff | feature         |   diff |
|:----------------|-------:|:----------------|-------:|:----------------|-------:|:----------------|-------:|
| e_acf10         |   0    | e_acf1         |   0    | diff2_acf1         |   0    | alpha         |   3.2    |
| seasonal_period |   0    | spike         |   0    | diff1_acf10         |   0    | arch_acf         |   3.3    |
| nperiods        |   0    | curvature         |   0    | x_acf1         |   0    | beta         |   4.04    |
| linearity       |   0    | crossing_points         |   0    | nonlinearity         |   0    | garch_r2         |   4.74    |
| hw_gamma        |   0    | lumpiness         |   0    | diff2x_pacf5         |   0    | hurst         |   5.45    |
| hw_beta         |   0    | diff1x_pacf5         |   0    | unitroot_kpss         |   0    | garch_acf         |   5.53    |
| hw_alpha        |   0    | diff1_acf10         |   0    | x_pacf5         |   0    | entropy         |   11.65    |
| trend           |   0    | arch_lm         |   0    | x_acf10         |   0    |
| flat_spots      |   0    | diff1_acf1         |   0    | unitroot_pp         |   0    |
| series_length   |   0    | stability         |   0    | arch_r2         |   1.37    |

To replicate this results use:

```
python -m tsfeatures.compare_with_r --results_directory ./data --dataset_name Daily --num_obs 100
```

### Sesonal data (100 Hourly M4 time series)

| feature           |   diff | feature      | diff | feature   | diff    | feature    | diff    |
|:------------------|-------:|:-------------|-----:|:----------|--------:|:-----------|--------:|
| series_length     |   0    |seas_acf1     | 0    | trend | 2.28 | hurst | 26.02 |
| flat_spots        |   0    |x_acf1|0| arch_r2 | 2.29 | hw_beta | 32.39 |
| nperiods          |   0    |unitroot_kpss|0| alpha | 2.52 | trough | 35 |
| crossing_points   |   0    |nonlinearity|0| beta | 3.67 | peak | 69 |
| seasonal_period   |   0    |diff1_acf10|0| linearity | 3.97 |
| lumpiness         |   0    |x_acf10|0| curvature | 4.8 |
| stability         |   0    |seas_pacf|0| e_acf10 | 7.05 |
| arch_lm           |   0    |unitroot_pp|0| garch_r2 | 7.32 |
| diff2_acf1        |   0    |spike|0| hw_gamma | 7.32 |
| diff2_acf10       |   0    |seasonal_strength|0.79| hw_alpha | 7.47 |
| diff1_acf1        |   0    |e_acf1|1.67| garch_acf | 7.53 |
| diff2x_pacf5      |   0    |arch_acf|2.18| entropy | 9.45 |

To replicate this results use:

```
python -m tsfeatures.compare_with_r --results_directory ./data --dataset_name Hourly --num_obs 100
```
