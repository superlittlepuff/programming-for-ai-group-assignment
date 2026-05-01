import numpy as np

class SortSearch:

    @staticmethod
    def stable_sort(
        arr: np.ndarray,
        axis: int = -1,
    ) -> np.ndarray:
        """
        Sort *arr* along *axis* using a stable algorithm

        Parameters
        ----------
        arr : array_like
            Input data.
        axis : int, default -1
            Axis along which to sort.  ``-1`` means the last axis.

        Returns
        -------
        sorted_arr : np.ndarray
            A **copy** of *arr* sorted along *axis*.  Shape is identical to
            the input.

        Raises
        ------
        np.AxisError
            If *axis* is out of range for the given array.
        ValueError
            If *arr* is 0-dimensional.

        Complexity
        ----------
        Time  : O(n log n)
        Space : O(n)

        Examples
        --------
        >>> SortSearch.stable_sort([3, 1, 2])
        array([1, 2, 3])
        >>> SortSearch.stable_sort([[3, 1], [4, 2]], axis=0)
        array([[3, 1],
               [4, 2]])
        """
        arr = np.asarray(arr)
        if arr.ndim == 0:
            raise ValueError("stable_sort does not support 0-dimensional arrays.")
        return np.sort(arr, kind='stable', axis=axis)
    
    @staticmethod
    def multi_key_sort(
        arr: np.ndarray,
        keys: list[int],
    ) -> np.ndarray:
        """
        Sort a 2-D array by multiple column indices (last key = highest priority).

        Uses ``np.lexsort``, which sorts by the *last* key in ``keys`` first
        (highest priority), then earlier keys as tiebreakers.

        Parameters
        ----------
        arr : array_like, shape (n, m)
            2-D input array.  Must have at least as many columns as
            ``max(keys) + 1``.
        keys : list[int]
            Column indices to sort by, ordered from **lowest** to **highest**
            priority.  Example: ``keys=[1, 0]`` sorts primarily by column 0
            and uses column 1 as the tiebreaker.

        Returns
        -------
        sorted_arr : np.ndarray, shape (n, m)
            Row-permuted copy of *arr*.

        Raises
        ------
        ValueError
            If *arr* is not 2-D, *keys* is empty, or a key index is out of
            range for the column dimension.

        Complexity
        ----------
        Time  : O(p · n log n) where p = len(keys)
        Space : O(n)
        """
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(
                f"multi_key_sort expects a 2-D array; got shape {arr.shape}."
            )
        if len(keys) == 0:
            raise ValueError("keys must contain at least one column index.")
        max_key = max(keys)
        if max_key >= arr.shape[1] or min(keys) < 0:
            raise ValueError(
                f"Key index {max_key} is out of bounds for array with "
                f"{arr.shape[1]} columns."
            )
        sorted_indices = np.lexsort([arr[:, key] for key in keys])
        return arr[sorted_indices]

    @staticmethod
    def topk(
        values: np.ndarray,
        k: int,
        largest: bool = True,
        return_indices: bool = True,
    ) -> np.ndarray:
        """
        Return the indices (or values) of the top-k elements

        Parameters
        ----------
        values : array_like, shape (n,)
            1-D input.
        k : int
            Number of elements to select.  Must satisfy ``1 <= k <= n``.
        largest : bool, default True
            If True, return the *largest* k elements; otherwise the *smallest*.
        return_indices : bool, default True
            If True, return the indices into *values*.
            If False, return the actual values of the top-k elements.

        Returns
        -------
        result : np.ndarray, shape (k,)
            Indices or values of the top-k elements in unspecified order.
            Call ``np.sort`` on the result if ordering within the k elements
            matters.

        Raises
        ------
        ValueError
            If *values* is not 1-D, k < 1, or k > n.

        Complexity
        ----------
        Time  : O(n) expected  (introselect in NumPy)
        Space : O(k)
        """
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(
                f"topk expects a 1-D array; got shape {values.shape}. "
                "Flatten first if needed."
            )
        n = len(values)
        if not (1 <= k <= n):
            raise ValueError(
                f"k={k} is out of bounds for array of length {n}. "
                f"k must satisfy 1 <= k <= {n}."
            )
        
        if largest:
            indices = np.argpartition(values, -k)[-k:]
        else:
            indices = np.argpartition(values, k)[:k]
        
        if return_indices:
            return indices
        else:
            return values[indices]
        
    def _partition(arr, left, right, pivot_index):
        pivot_value = arr[pivot_index]
        arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
        store_index = left
        
        for i in range(left, right):
            if arr[i] < pivot_value:
                arr[store_index], arr[i] = arr[i], arr[store_index]
                store_index += 1
        
        arr[right], arr[store_index] = arr[store_index], arr[right]
        return store_index
    
    @staticmethod
    def kth_smallest(
        arr: np.ndarray,
        k: int,
        left: int = 0,
        right: int | None = None,
    ) -> float:
        """
        Quickselect — find the k-th smallest element of arr
        in the subarray ``arr[left:right+1]``.

        Parameters
        ----------
        arr : array_like
            1-D numeric input.
        k : int
            1-based rank within ``arr[left:right+1]``.  ``k=1`` → minimum,
            ``k=right-left+1`` → maximum of the window.
        left : int, default 0
            Left boundary of the active window (inclusive).
        right : int or None, default None
            Right boundary (inclusive).  Defaults to ``len(arr) - 1``.

        Returns
        -------
        value : float
            The k-th smallest element in the window.

        Raises
        ------
        ValueError
            If arr is not 1-D or k is out of the valid range
            ``[1, right - left + 1]``.

        Complexity
        ----------
        Time  : O(n) expected, O(n²) worst case (randomised pivot)
        Space : O(log n)
        """
        if right is None:
            right = len(arr) - 1
        if k <= 0 or k > right - left + 1:
            raise ValueError("k is out of bounds")
        arr = np.asarray(arr)

        pivot_index = np.random.randint(left, right + 1)
        pivot_index = SortSearch._partition(arr, left, right, pivot_index)

        if pivot_index == left + k - 1:
            return arr[pivot_index]
        elif pivot_index > left + k - 1:
            return SortSearch.kth_smallest(arr, k, left, pivot_index - 1)
        else:
            return SortSearch.kth_smallest(arr, k - (pivot_index - left + 1), pivot_index + 1, right)

    @staticmethod
    def binary_search(
        sorted_array: np.ndarray,
        x: float,
    ) -> tuple[int, bool]:
        """
        Finds the leftmost position where *x* could be inserted to keep the
        array sorted.

        Parameters
        ----------
        sorted_array : array_like, shape (n,)
            A 1-D array sorted in ascending order.  Behaviour is undefined
            for unsorted input.
        x : scalar
            The value to search for.

        Returns
        -------
        index : int
            Insertion index ``i`` such that
            ``sorted_array[i-1] < x <= sorted_array[i]``.
            Range: ``[0, n]``.
        exists : bool
            True if and only if ``sorted_array[index] == x``.

        Raises
        ------
        ValueError
            If sorted_array is not 1-D.

        Complexity
        ----------
        Time  : O(log n)
        Space : O(1)
        """
        sorted_array = np.asarray(sorted_array)
        if sorted_array.ndim != 1:
            raise ValueError(
                f"binary_search expects a 1-D array; got shape {sorted_array.shape}."
            )
        
        left, right = 0, len(sorted_array)

        while left < right:
            mid = (left + right) // 2
            if sorted_array[mid] < x:
                left = mid + 1
            else:
                right = mid
        exists = (left < len(sorted_array) and sorted_array[left] == x)
        return left, exists