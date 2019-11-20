from tsfeatures.funcs import *

def tsfeatures(tslist,
              features = [stl_features, frequency, entropy, acf_features],
              scale = True):
    if scale:
        tslist = [scalets(ts) for ts in tslist]
    
    feat_df = pd.concat(
        [
            pd.DataFrame(
                dict(
                    ChainMap(*[dict_feat for dict_feat in [func(ts) for func in features]])
                ),
                index = [0]
            ) for ts in tslist
        ]
    ).reset_index(drop=True)
    
    return feat_df