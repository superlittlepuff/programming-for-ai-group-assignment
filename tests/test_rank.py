import unittest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute.rank import Rank

class TestRank(unittest.TestCase):
    """Tests for Rank.rank."""

    def test_average_no_ties(self):
        data = np.array([10.0, 30.0, 20.0])
        result = Rank.rank(data, method="average")
        np.testing.assert_array_equal(result, [1.0, 3.0, 2.0])

    def test_average_with_ties(self):
        result = Rank.rank([1.0, 1.0, 3.0], method="average")
        np.testing.assert_allclose(result, [1.5, 1.5, 3.0])
        result2 = Rank.rank([3, 1, 4, 1, 5], method="average")
        np.testing.assert_allclose(result2, [3.0, 1.5, 4.0, 1.5, 5.0])

    def test_dense_with_ties(self):
        result = Rank.rank([3, 1, 4, 1, 5], method="dense")
        np.testing.assert_array_equal(result, [2, 1, 3, 1, 4])

    def test_ordinal_ties_by_position(self):
        result = Rank.rank([1, 1, 3], method="ordinal")
        np.testing.assert_array_equal(result, [1, 2, 3])
        result2 = Rank.rank([3, 1, 4, 1, 5], method="ordinal")
        np.testing.assert_array_equal(result2, [3, 1, 4, 2, 5])

    def test_all_equal_average(self):
        data = [5.0, 5.0, 5.0, 5.0]
        result = Rank.rank(data, method="average")
        np.testing.assert_array_equal(result, [2.5, 2.5, 2.5, 2.5])

    def test_single_element(self):
        result = Rank.rank([42.0], method="average")
        np.testing.assert_array_equal(result, [1.0])

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            Rank.rank([1, 2, 3], method="bogus")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            Rank.rank([], method="average")

    def test_2d_raises(self):
        with self.assertRaises(ValueError):
            Rank.rank(np.array([[1, 2], [3, 4]]))
    
    def test_nan_handling(self):
        with self.assertRaises(ValueError):
            Rank.rank(np.array([1.0, float("nan"), 3.0]), method="average")


class TestPercentile(unittest.TestCase):
    """Tests for Rank.percentile."""

    def test_median(self):
        self.assertAlmostEqual(Rank.percentile([1, 2, 3, 4, 5], 50), 3.0)

    def test_0th_and_100th(self):
        self.assertAlmostEqual(Rank.percentile([1, 2, 3], 0), 1.0)
        self.assertAlmostEqual(Rank.percentile([1, 2, 3], 100), 3.0)

    def test_multiple_quantiles(self):
        result = Rank.percentile([1, 2, 3, 4, 5], [0, 25, 50, 100])
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 5.0])

    def test_nan_ignored(self):
        result = Rank.percentile([1.0, float("nan"), 3.0, 5.0], 50)
        self.assertAlmostEqual(result, 3.0)
        result2 = Rank.percentile([1, 2, float('nan'), 4, 5], 50)
        self.assertAlmostEqual(result2, 3.0)

    def test_lower_interpolation(self):
        result = Rank.percentile([1, 2, 3, 4], 25, interpolation="lower")
        self.assertEqual(result, 1.0)

    def test_higher_interpolation(self):
        result = Rank.percentile([1, 2, 3, 4], 25, interpolation="higher")
        self.assertEqual(result, 2.0)

    def test_midpoint_interpolation(self):
        result = Rank.percentile([1, 2, 3, 4], 25, interpolation="midpoint")
        self.assertAlmostEqual(result, 1.5)

    def test_invalid_q_raises(self):
        with self.assertRaises(ValueError):
            Rank.percentile([1, 2, 3], -1)
        with self.assertRaises(ValueError):
            Rank.percentile([1, 2, 3], 101)

    def test_invalid_interpolation_raises(self):
        with self.assertRaises(ValueError):
            Rank.percentile([1, 2, 3], 50, interpolation="cubic")

    def test_all_nan_raises(self):
        with self.assertRaises(ValueError):
            Rank.percentile([float("nan"), float("nan")], 50)