import numpy as np

class SortSearch:
    def stable_sort(arr, axis=-1):
        arr = np.asarray(arr)
        return np.sort(arr, kind='stable', axis=axis)
    
    def multi_key_sort(arr, keys):
        arr = np.asarray(arr)
        sorted_indices = np.lexsort([arr[:, key] for key in reversed(keys)])
        return arr[sorted_indices]

    def topk(values, k, largest=True, return_indices=True):
        values = np.asarray(values)
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
    
    # Quickselect algorithm to find the k-th smallest element with k starting from 1
    # and k is between the range rather than 0 and len(arr)
    # left and right are inclusive
    def kth_element(arr: np.ndarray, k, left=0, right=None, largest=True):
        if right is None:
            right = len(arr) - 1
        if k <= 0 or k > right - left + 1:
            raise ValueError("k is out of bounds")
        arr = np.asarray(arr)
        if largest:
            arr = -arr
        
        pivot_index = np.random.randint(left, right + 1)
        pivot_index = SortSearch._partition(arr, left, right, pivot_index)

        if pivot_index == left + k - 1:
            return arr[pivot_index]
        elif pivot_index > left + k - 1:
            return SortSearch.kth_element(arr, k, left, pivot_index - 1)
        else:
            return SortSearch.kth_element(arr, k - (pivot_index - left + 1), pivot_index + 1, right)

    def binary_search(sorted_array, x):
        left, right = 0, len(sorted_array)
        while left < right:
            mid = (left + right) // 2
            if sorted_array[mid] < x:
                left = mid + 1
            else:
                right = mid
        exists = (left < len(sorted_array) and sorted_array[left] == x)
        return left, exists