import numpy as np

class Rank:
    def rank(data, method='average'):
        if method == 'average':
            return np.argsort(np.argsort(data))
        elif method == 'dense':
            unique, inverse = np.unique(data, return_inverse=True)
            return inverse
        elif method == 'ordinal':
            return np.argsort(np.argsort(data)) + 1
        else:
            raise ValueError("Invalid method")

    def percentile(data, q, interpolation='linear'):
        if interpolation == 'linear':
            return np.percentile(data, q)
        elif interpolation == 'lower':
            return np.percentile(data, q, interpolation='lower')
        elif interpolation == 'higher':
            return np.percentile(data, q, interpolation='higher')
        elif interpolation == 'midpoint':
            return np.percentile(data, q, interpolation='midpoint')
        else:
            raise ValueError("Invalid interpolation method")
        
a = [3, 1, 4, 1, 5]
print(Rank.rank(a, method='average'))