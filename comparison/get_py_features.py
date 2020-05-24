import pandas as pd

from tsfeatures import tsfeatures

series = pd.read_csv('sample_series.csv')

features = tsfeatures(series, 7)
features.index.name = 'unique_id'
features.to_csv('pyfeatures.csv')
