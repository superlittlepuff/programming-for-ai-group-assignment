import numpy as np

class Stats:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=float)

    def mean(self):
        """
        Arithmetic mean, ignoring NaNs.

        Returns
        -------
        float
            Arithmetic mean of array

        Raises
        ------
        ValueError
            If all elements are NaN.

        Complexity : O(n) time, O(1).
        """
        self._check_not_all_nan()
        return float(np.nanmean(self.arr))

    def median(self):
        """
        Median, ignoring NaNs.

        Returns
        -------
        float
            Middle value (or average of the two middle values) of the
            sorted non-NaN elements.

        Raises
        ------
        ValueError
            If all elements are NaN.

        Complexity : O(n) time, O(n) space.
        """
        self._check_not_all_nan()
        return float(np.nanmedian(self.arr))

    def std(self) -> float:
        """
        Standard deviation, ignoring NaNs.
 
        Returns
        -------
        float
            ``np.nanstd(self.arr)``
 
        Raises
        ------
        ValueError
            If all elements are NaN.
 
        Complexity : O(n) time, O(1).
        """
        self._check_not_all_nan()
        return float(np.nanstd(self.arr))

    def min(self) -> float:
        """
        Minimum value, ignoring NaNs.
 
        Returns
        -------
        float
 
        Raises
        ------
        ValueError
            If all elements are NaN.
 
        Complexity : O(n) time, O(1) space.
        """
        self._check_not_all_nan()
        return float(np.nanmin(self.arr))

    def max(self) -> float:
        """
        Maximum value, ignoring NaNs.
 
        Returns
        -------
        float
 
        Raises
        ------
        ValueError
            If all elements are NaN.
 
        Complexity : O(n) time, O(1) space.
        """
        self._check_not_all_nan()
        return float(np.nanmax(self.arr))

    def histogram(
        self,
        bins: int | np.ndarray = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a histogram of the array, ignoring NaNs.
 
        Parameters
        ----------
        bins : int or array_like, default 10
            If int, the number of equal-width bins.
            If array, the bin edge sequence.
 
        Returns
        -------
        counts : np.ndarray
            Bin counts.
        bin_edges : np.ndarray
            Bin edge values.
 
        Raises
        ------
        ValueError
            If all elements are NaN or bins is invalid.
 
        Complexity
        ----------
        Time  : O(n log k) where k = number of bins
        Space : O(n + k)
        """
        clean = self.arr.ravel()
        clean = clean[~np.isnan(clean)]
        if clean.size == 0:
            raise ValueError("histogram requires at least one non-NaN element.")
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
    
    def _check_not_all_nan(self):
        if np.isnan(self.arr).all():
            raise ValueError("All elements are NaN; cannot compute statistic.")