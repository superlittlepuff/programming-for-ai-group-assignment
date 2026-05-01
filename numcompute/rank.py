import numpy as np

class Rank:
    """
    Ranking and percentile utilities for one-dimensional numeric arrays.

    This class provides ranking methods with different tie-handling
    strategies as well as percentile estimation.
    """

    def rank(data: np.ndarray, method='average'):
        data = np.asarray(data)
        if method == 'average':
            sorted_indices = np.argsort(data)
            sorted_data = data[sorted_indices]
            ranks = np.zeros_like(data, dtype=float)
            unique, inverse, counts = np.unique(sorted_data, return_inverse=True, return_counts=True)
            cumulative_counts = np.cumsum(counts)
            for i in range(len(unique)):
                start = cumulative_counts[i - 1] if i > 0 else 0
                end = cumulative_counts[i]
                ranks[sorted_indices[start:end]] = (start + end + 1) / 2.0
            return ranks
        elif method == 'dense':
            unique, inverse = np.unique(data, return_inverse=True)
            return inverse + 1
        elif method == 'ordinal':
            return data.argsort().argsort() + 1
        else:
            raise ValueError("Invalid method")

    def percentile(data, q, interpolation='linear'):
        if interpolation == 'linear':
            return np.percentile(data, q)
        elif interpolation == 'lower':
            return np.percentile(data, q, method='lower')
        elif interpolation == 'higher':
            return np.percentile(data, q, method='higher')
        elif interpolation == 'midpoint':
            return np.percentile(data, q, method='midpoint')
        else:
            raise ValueError("Invalid interpolation method")

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("Average Rank:", Rank.rank(arr, method='average'))
print("Dense Rank:", Rank.rank(arr, method='dense'))
print("Ordinal Rank:", Rank.rank(arr, method='ordinal'))
print("25th Percentile (linear):", Rank.percentile(arr, 25, interpolation='linear'))
print("25th Percentile (lower):", Rank.percentile(arr, 25, interpolation='lower'))
print("25th Percentile (higher):", Rank.percentile(arr, 25, interpolation='higher'))
print("25th Percentile (midpoint):", Rank.percentile(arr, 25, interpolation='midpoint'))
