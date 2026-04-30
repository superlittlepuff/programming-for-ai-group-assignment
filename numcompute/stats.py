import numpy as np

class Stats:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def mean(self):
        return np.nanmean(self.arr)

    def median(self):
        return np.nanmedian(self.arr)

    def std(self):
        return np.nanstd(self.arr)

    def min(self):
        return np.nanmin(self.arr)

    def max(self):
        return np.nanmax(self.arr)

    def histogram(self, bins=10):
        clean = self.arr.ravel()
        clean = clean[~np.isnan(clean)]
        return np.histogram(clean, bins=bins)

    def quantiles(self, q):
        return np.nanquantile(self.arr, q)

    def axis_stats(self, axis=None):
        return {
            'mean': np.nanmean(self.arr, axis=axis),
            'median': np.nanmedian(self.arr, axis=axis),
            'std': np.nanstd(self.arr, axis=axis),
            'min': np.nanmin(self.arr, axis=axis),
            'max': np.nanmax(self.arr, axis=axis)
        }
