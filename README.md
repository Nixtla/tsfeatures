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

This package receives a list of time series and a frecuency.

``` python
series = []
frcy = 7
tsfeatures(series, frcy=7)
```

## Parallel!


``` python
tsfeatures(series, frcy=7, parallel=True)
```

