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

## Comparison with the R implementation

| feature           | diff  |
|:------------------|------:|
| series_length     |  0    |
| trough            |  0    |
| peak              |  0    |
| nperiods          |  0    |
| seasonal_period   |  0    |
| crossing_points   |  0    |
| spike             |  0    |
| arch_lm           |  0    |
| stability         |  0    |
| seasonal_strength |  0    |
| e_acf1            |  0    |
| trend             |  0    |
| e_acf10           |  0    |
| x_pacf5           |  0    |
| seas_pacf         |  0    |
| seas_acf1         |  0    |
| x_acf1            |  0    |
| lumpiness         |  0    |
| diff1_acf1        |  0    |
| diff1_acf10       |  0    |
| nonlinearity      |  0    |
| x_acf10           |  0    |
| diff2_acf1        |  0    |
| diff2_acf10       |  0    |
| diff1x_pacf5      |  0    |
| unitroot_kpss     |  0    |
| curvature         |  0    |
| linearity         |  0    |
| diff2x_pacf5      |  0    |
| unitroot_pp       |  0    |
| arch_acf          |  0.24 |
| arch_r2           |  0.25 |
| hw_alpha          |  1.88 |
| hw_beta           |  1.99 |
| flat_spots        |  2.00 |
| entropy           |  2.33 |
| garch_r2          |  2.51 |
| garch_acf         |  3.00 |
| alpha             |  3.42 |
| beta              |  3.78 |
| hw_gamma          |  4.18 |
| hurst             | 12.41 |
