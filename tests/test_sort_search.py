import unittest
import numpy as np
import sys, os
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
from numcompute.sort_search import SortSearch

class TestStableSort(unittest.TestCase):
    def test_basic_1d(self):
        result = SortSearch.stable_sort([3, 1, 4, 1, 5, 9, 2, 6])
        np.testing.assert_array_equal(result, [1, 1, 2, 3, 4, 5, 6, 9])

    def test_already_sorted(self):
        arr = [1, 2, 3, 4, 5]
        np.testing.assert_array_equal(SortSearch.stable_sort(arr), arr)

    def test_reverse_sorted(self):
        np.testing.assert_array_equal(
            SortSearch.stable_sort([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5]
        )

    def test_single_element(self):
        np.testing.assert_array_equal(SortSearch.stable_sort([42]), [42])

    def test_all_equal(self):
        np.testing.assert_array_equal(
            SortSearch.stable_sort([7, 7, 7, 7]), [7, 7, 7, 7]
        )

    def test_2d_axis0(self):
        arr = np.array([[3, 1], [1, 4]])
        result = SortSearch.stable_sort(arr, axis=0)
        np.testing.assert_array_equal(result, [[1, 1], [3, 4]])

    def test_2d_axis1(self):
        arr = np.array([[3, 1], [4, 2]])
        result = SortSearch.stable_sort(arr, axis=1)
        np.testing.assert_array_equal(result, [[1, 3], [2, 4]])

    def test_0d_raises(self):
        with self.assertRaises(ValueError):
            SortSearch.stable_sort(np.array(5))

    def test_stability_preserves_order_of_equal_keys(self):
        arr = np.array([2, 1, 2, 1])
        sorted_arr = SortSearch.stable_sort(arr)
        np.testing.assert_array_equal(sorted_arr, [1, 1, 2, 2])


class TestMultiKeySort(unittest.TestCase):
    def test_sort_by_single_column(self):
        arr = np.array([[3, 2], [1, 4], [2, 1]])
        result = SortSearch.multi_key_sort(arr, keys=[0])
        np.testing.assert_array_equal(result[:, 0], [1, 2, 3])

    def test_sort_by_two_columns_tiebreak(self):
        arr = np.array([[1, 3], [1, 1], [2, 2]])
        result = SortSearch.multi_key_sort(arr, keys=[0, 1])
        np.testing.assert_array_equal(result, [[1, 1], [2, 2], [1, 3]])
        result2 = SortSearch.multi_key_sort(arr, keys=[1, 0])
        np.testing.assert_array_equal(result2, [[1, 1], [1, 3], [2, 2]])

    def test_not_2d_raises(self):
        with self.assertRaises(ValueError):
            SortSearch.multi_key_sort([1, 2, 3], keys=[0])

    def test_empty_keys_raises(self):
        with self.assertRaises(ValueError):
            SortSearch.multi_key_sort(np.array([[1, 2]]), keys=[])

    def test_key_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            SortSearch.multi_key_sort(np.array([[1, 2]]), keys=[5])


class TestTopK(unittest.TestCase):
    def test_largest_indices(self):
        vals = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        indices = SortSearch.topk(vals, k=3, largest=True, return_indices=True)
        self.assertEqual(set(indices), {4, 5, 7})

    def test_smallest_indices(self):
        vals = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        indices = SortSearch.topk(vals, k=2, largest=False, return_indices=True)
        self.assertEqual(set(indices), {1, 3})

    def test_return_values(self):
        vals = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = SortSearch.topk(vals, k=3, largest=True, return_indices=False)
        self.assertEqual(set(result), {5, 9, 6})

    def test_k_equals_n(self):
        vals = np.array([1, 2, 3])
        indices = SortSearch.topk(vals, k=3, largest=True, return_indices=True)
        self.assertEqual(len(indices), 3)

    def test_k_exceeds_n_raises(self):
        with self.assertRaises(ValueError):
            SortSearch.topk([1, 2, 3], k=5)

    def test_k_zero_raises(self):
        with self.assertRaises(ValueError):
            SortSearch.topk([1, 2, 3], k=0)

    def test_2d_raises(self):
        with self.assertRaises(ValueError):
            SortSearch.topk(np.array([[1, 2], [3, 4]]), k=1)


class TestKthSmallest(unittest.TestCase):
    def test_kth_smallest(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(SortSearch.kth_smallest(arr, k=3), 2)

    def test_k_equals_1(self):
        arr = np.array([3, 1, 4])
        self.assertEqual(SortSearch.kth_smallest(arr, k=1), 1)

    def test_k_equals_n(self):
        arr = np.array([3, 1, 4])
        self.assertEqual(SortSearch.kth_smallest(arr, k=3), 4)

    def test_kth_smallest_with_window(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(SortSearch.kth_smallest(arr, k=2, left=2, right=5), 4)
    
    def test_kth_smallest_with_window_k_equals_window_size(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(SortSearch.kth_smallest(arr, k=4, left=2, right=5), 9)

    def test_kth_smallest_with_window_k_exceeds_window_size_raises(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        with self.assertRaises(ValueError):
            SortSearch.kth_smallest(arr, k=5, left=2, right=5)

    def test_kth_smallest_window_equal_one(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(SortSearch.kth_smallest(arr, k=1, left=5, right=5), 9)

    def test_k_exceeds_n_raises(self):
        arr = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            SortSearch.kth_smallest(arr, k=4)

    def test_k_zero_raises(self):
        arr = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            SortSearch.kth_smallest(arr, k=0)

    def test_2d_raises(self):
        with self.assertRaises(ValueError):
            SortSearch.kth_smallest(np.array([[1, 2], [3, 4]]), k=1)
        


class TestBinarySearch(unittest.TestCase):
    def test_found(self):
        idx, exists = SortSearch.binary_search([1, 3, 5, 7, 9], 5)
        self.assertEqual(idx, 2)
        self.assertTrue(exists)

    def test_not_found_insertion_middle(self):
        idx, exists = SortSearch.binary_search([1, 3, 5, 7], 4)
        self.assertEqual(idx, 2)
        self.assertFalse(exists)

    def test_not_found_before_start(self):
        idx, exists = SortSearch.binary_search([1, 3, 5], 0)
        self.assertEqual(idx, 0)
        self.assertFalse(exists)

    def test_not_found_after_end(self):
        idx, exists = SortSearch.binary_search([1, 3, 5], 10)
        self.assertEqual(idx, 3)
        self.assertFalse(exists)

    def test_empty_array(self):
        idx, exists = SortSearch.binary_search([], 1)
        self.assertEqual(idx, 0)
        self.assertFalse(exists)

    def test_single_element_found(self):
        idx, exists = SortSearch.binary_search([7], 7)
        self.assertEqual(idx, 0)
        self.assertTrue(exists)

    def test_duplicate_elements_leftmost(self):
        idx, exists = SortSearch.binary_search([1, 2, 2, 2, 3], 2)
        self.assertEqual(idx, 1)
        self.assertTrue(exists)

    def test_2d_raises(self):
        with self.assertRaises(ValueError):
            SortSearch.binary_search(np.array([[1, 2], [3, 4]]), 1)