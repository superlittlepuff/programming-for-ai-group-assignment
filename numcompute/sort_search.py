import numpy as np

class SortSearch:
    def stable_sort(arr):
        return np.sort(arr, kind='stable')

    def multi_key_sort(arr, keys):
        return arr[np.lexsort(keys)]

    def topk(values, k, largest=True, return_indices=True):
        if largest:
            indices = np.argpartition(values, -k)[-k:]
        else:
            indices = np.argpartition(values, k)[:k]
        
        if return_indices:
            return indices
        else:
            return values[indices]

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