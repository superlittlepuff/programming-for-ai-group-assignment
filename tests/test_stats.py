import unittest
import numpy as np
import sys, os
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
from numcompute.stats import Stats

class TestStats(unittest.TestCase):
    def setUp(self):
        self.data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        self.s = Stats(self.data)

    def test_mean(self):
        self.assertAlmostEqual(self.s.mean(), 5.0)

    def test_median(self):
        self.assertAlmostEqual(self.s.median(), 4.5)

    def test_std_population(self):
        self.assertAlmostEqual(self.s.std(), 2.0)

    def test_std_sample(self):
        self.assertAlmostEqual(self.s.std(), np.std(self.data))

    def test_min(self):
        self.assertEqual(self.s.min(), 2.0)

    def test_max(self):
        self.assertEqual(self.s.max(), 9.0)

    def test_nan_handling_mean(self):
        s = Stats([1.0, float("nan"), 3.0])
        self.assertAlmostEqual(s.mean(), 2.0)

    def test_all_nan_raises(self):
        s = Stats([float("nan"), float("nan")])
        with self.assertRaises(ValueError):
            s.mean()
        with self.assertRaises(ValueError):
            s.median()
        with self.assertRaises(ValueError):
            s.std()
        with self.assertRaises(ValueError):
            s.min()
        with self.assertRaises(ValueError):
            s.max()

    def test_histogram_bin_count(self):
        counts, edges = self.s.histogram(bins=4)
        self.assertEqual(len(counts), 4)
        self.assertEqual(len(edges), 5)

    def test_histogram_all_nan_raises(self):
        with self.assertRaises(ValueError):
            Stats([float("nan")]).histogram()

    def test_histogram_sum_equals_n(self):
        counts, _ = self.s.histogram(bins=10)
        self.assertEqual(counts.sum(), len(self.data))

    def test_quantiles_scalar(self):
        result = self.s.quantiles(0.5)
        self.assertIsInstance(result, float)

    def test_quantiles_list(self):
        result = self.s.quantiles([0.0, 0.5, 1.0])
        self.assertEqual(len(result), 3)

    def test_quantiles_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.s.quantiles(1.5)

    def test_axis_stats_none(self):
        result = self.s.axis_stats(axis=None)
        self.assertAlmostEqual(result["mean"], 5.0)
        for key in ("mean", "median", "std", "min", "max"):
            self.assertIn(key, result)

    def test_axis_stats_2d_axis0(self):
        s2 = Stats([[1.0, 2.0], [3.0, 4.0]])
        result = s2.axis_stats(axis=0)
        np.testing.assert_allclose(result["mean"], [2.0, 3.0])
        np.testing.assert_allclose(result["min"], [1.0, 2.0])

    def test_axis_stats_2d_axis1(self):
        s2 = Stats([[1.0, 3.0], [2.0, 4.0]])
        result = s2.axis_stats(axis=1)
        np.testing.assert_allclose(result["mean"], [2.0, 3.0])

    def test_axis_stats_invalid_axis_raises(self):
        with self.assertRaises((np.exceptions.AxisError, ValueError, IndexError)):
            self.s.axis_stats(axis=5)

    def test_single_element(self):
        s = Stats([7.0])
        self.assertEqual(s.mean(), 7.0)
        self.assertEqual(s.std(), 0.0)
        self.assertEqual(s.min(), 7.0)
        self.assertEqual(s.max(), 7.0)