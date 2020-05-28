import pandas as pd
import subprocess

from pathlib import Path

PATH = Path('.')

if not (PATH/'Rfeatures.csv').exists():
    subprocess.run('Rscript get_R_features.R'.split())

if not (PATH/'pyfeatures.csv').exists():
    subprocess.run('python get_py_features.py'.split())

py_features = pd.read_csv('pyfeatures.csv', index_col=['unique_id'])
R_features = pd.read_csv('Rfeatures.csv', index_col=['unique_id'])

R_features = R_features.rename(columns={'length': 'series_length',
                                        'ARCH.LM': 'arch_lm'})
cols = py_features.columns
R_features = R_features[cols]

print('Sums of absolute differences',
      (py_features - R_features).abs().sum(0).sort_values(),
      sep='\n')
