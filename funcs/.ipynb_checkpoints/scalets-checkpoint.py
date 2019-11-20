def scalets(x):
    scaledx = scale(x, axis=0, with_mean=True, with_std=True, copy=True)
    ts = pd.Series(scaledx, index=x.index)
    return ts
