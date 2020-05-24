# tsfeatures

This library replicates _[tsfeatures](https://github.com/robjhyndman/tsfeatures)_, R package.


# Install

``` python
pip install git+https://github.com/FedericoGarza/tsfeatures
```


# Use

``` python
from tsfeatures import tsfeatures
```

This package receives a panel pandas df with columns `unique_id`, `ds`, `y`.

``` python
tsfeatures(panel, freq=7)
```

## Parallel!


``` python
tsfeatures(panel, freq=7, parallel=True)
```
