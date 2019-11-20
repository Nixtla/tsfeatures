def entropy(x):
    try:
        # Maybe 100 can change
        entropy = spectral_entropy(x, 1)
    except:
        entropy = None
        
    return {'entropy': entropy}

class Entropy():
    pass