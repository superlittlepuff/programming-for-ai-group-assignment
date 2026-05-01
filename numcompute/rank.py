import numpy as np

class Rank:
    @staticmethod
    def rank(
        data: np.ndarray,
        method: str = "average",
    ) -> np.ndarray:
        """
        Assign ranks to a 1-D array, with configurable tie-breaking.

        Parameters
        ----------
        data : array_like, shape (n,)
            Input values.  Must be 1-D and contain at least one element.
            NaN values are not supported and will produce undefined results;
            filter them before calling.
        method : {'average', 'dense', 'ordinal'}
            How to handle ties:
 
            * 'average'  — tied elements share the mean of their ranks.
              E.g., values ``[1, 1, 3]`` → ranks ``[1.5, 1.5, 3.0]``.
            * 'dense'    — tied elements share the lowest rank; ranks are
              consecutive integers with no gaps.
              E.g., ``[1, 1, 3]`` → ``[1, 1, 2]``.
            * 'ordinal'  — ties broken by order of appearance (first-seen
              gets the lower rank). E.g., ``[1, 1, 3]`` → ``[1, 2, 3]``.

        Returns
        -------
        ranks : np.ndarray, shape (n,),

        Raises
        ------
        ValueError
            If *data* is not 1-D, is empty, contains NaN values, 
            or *method* is not one of the recognised strings.

        Complexity
        ----------
        Time  : O(n log n)  — used ``np.argsort`` / ``np.unique``
        Space : O(n)
        """
        data = np.asarray(data, dtype=float)
        if np.isnan(data).any():
            raise ValueError("NaN values are not supported in rank.")
        if data.ndim != 1:
            raise ValueError(
                f"rank expects a 1-D array; got shape {data.shape}. "
                "Flatten first if needed."
            )
        if data.size == 0:
            raise ValueError("rank requires at least one element.")
        
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
            return data.argsort(kind='stable').argsort() + 1
        else:
            raise ValueError("Invalid method. Must be 'average', 'dense', or 'ordinal'.")

    @staticmethod
    def percentile(
        data: np.ndarray,
        q: float | list[float],
        interpolation: str = "linear",
    ) -> float | np.ndarray:
        """
        Compute percentile(s) of a 1-D dataset.

        A wrapper around ``np.percentile`` that validates inputs and
        maps the *interpolation* keyword to NumPy's ``method`` parameter.

        Parameters
        ----------
        data : array_like, shape (n,)
            Input data.  NaN values are ignored
        q : float or list of float
            Percentile(s) to compute, in the range ``[0, 100]``.
        interpolation : {'linear', 'lower', 'higher', 'midpoint'}
            Controls how percentiles are interpolated between data points:

            * ``'linear'``   — default; weighted average of adjacent ranks.
            * ``'lower'``    — lower of the two surrounding values.
            * ``'higher'``   — higher of the two surrounding values.
            * ``'midpoint'`` — average of lower and higher.

        Returns
        -------
        result : float or np.ndarray
            Scalar if *q* is a scalar; 1-D array of the same length as *q*
            otherwise.

        Raises
        ------
        ValueError
            If *data* is not 1-D, is empty (after dropping NaNs), *q* is
            outside ``[0, 100]``, or *interpolation* is unrecognised.

        Complexity
        ----------
        Time  : O(n)  — NumPy uses introselect for non-linear methods
        Space : O(n)
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise ValueError(
                f"percentile expects a 1-D array; got shape {data.shape}."
            )
        clean = data[~np.isnan(data)]
        if clean.size == 0:
            raise ValueError(
                "percentile requires at least one non-NaN element."
            )
        q_arr = np.asarray(q, dtype=float)
        if np.any(q_arr < 0) or np.any(q_arr > 100):
            raise ValueError(
                f"All percentile values must be in [0, 100]; got {q}."
            )
        if interpolation == 'linear':
            return np.percentile(clean, q_arr)
        elif interpolation == 'lower':
            return np.percentile(clean, q_arr, method='lower')
        elif interpolation == 'higher':
            return np.percentile(clean, q_arr, method='higher')
        elif interpolation == 'midpoint':
            return np.percentile(clean, q_arr, method='midpoint')
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
