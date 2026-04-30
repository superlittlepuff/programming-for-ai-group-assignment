import unittest
import numpy as np
import os
from numcompute.io import load_csv
from numcompute.preprocessing import StandardScaler, SimpleImputer, OneHotEncoder

class TestNumCompute(unittest.TestCase):
    # --- 1. Data IO test (Load CSV) ---
    def test_load_csv_basic(self):
        """Test loading CSV successfully"""
        data = load_csv("student_performance_data.csv")
        self.assertEqual(data.shape, (30, 9))
        self.assertTrue(np.any(data == 'nan')) # Ensured identification NaN

    def test_load_csv_file_not_found(self):
        """Test file not found error"""
        with self.assertRaises(FileNotFoundError):
            load_csv("non_existent.csv")

    # --- 2. SimpleImputer test (missing value filling) ---
    def test_imputer_constant_fill(self):
        """Test constant filling NaN"""
        X = [[1, np.nan], [np.nan, 3]]
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        out = imputer.fit_transform(X)
        self.assertEqual(out[0, 1], 0)
        self.assertEqual(out[1, 0], 0)

    def test_imputer_empty_array(self):
        """Edge case: Empty array error"""
        imputer = SimpleImputer()
        with self.assertRaises(ValueError):
            imputer.fit(np.array([[]]))

    # --- 3. StandardScaler Test (Numerical Stability and Scaling) ---
    def test_scaler_standard_logic(self):
        """Testing Standardized Mathematical Logic"""
        X = np.array([[1.0], [2.0], [3.0]])
        scaler = StandardScaler()
        out = scaler.fit_transform(X)
        self.assertAlmostEqual(np.mean(out), 0, places=7)
        self.assertAlmostEqual(np.std(out), 1, places=7)

    def test_scaler_all_equal_values(self):
        """Edge cases: Handle all equal values ​​(prevent division by zero)"""
        X = np.array([[5.0, 5.0], [5.0, 5.0]])
        scaler = StandardScaler()
        out = scaler.fit_transform(X)
        # If the standard deviation is 0, it should remain unchanged and not collapse (the scale becomes 1).
        self.assertTrue(np.all(out == 0))

    def test_scaler_with_nans(self):
        """The test ignores NaN and calculates the mean/standard deviation."""
        X = np.array([[1.0], [np.nan], [3.0]])
        scaler = StandardScaler()
        scaler.fit(X)
        # (1+3)/2 = 2.0
        self.assertEqual(scaler.mean_, 2.0)

    def test_scaler_non_contiguous(self):
        """Edge cases: Non-contiguous memory strides"""
        # Create a non-contiguous memory array by transposing.
        X = np.random.rand(10, 2).T 
        scaler = StandardScaler()
        try:
            scaler.fit(X)
        except Exception as e:
            self.fail(f"StandardScaler failed on non-contiguous array: {e}")

    # --- 4. OneHotEncoder test (Category conversion) ---
    def test_onehot_basic(self):
        """Test one-hot encoding conversion"""
        X = np.array([['CS'], ['IT'], ['CS']])
        encoder = OneHotEncoder()
        out = encoder.fit_transform(X)
        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(np.sum(out), 3) # Each line should contain only one 1.

    def test_onehot_unseen_category(self):
        """The test encountered a class that was not seen during training when transforming."""
        encoder = OneHotEncoder()
        encoder.fit([['A'], ['B']])
        with self.assertRaises(ValueError):
            encoder.transform([['C']])

    # --- 5. Comprehensive stability and anomaly testing ---
    def test_not_fitted_error(self):
        """Tests are called when not fitted. transform"""
        scaler = StandardScaler()
        with self.assertRaises(ValueError):
            scaler.transform([[1, 2]])

    def test_extreme_values(self):
        """Test the maxima and minima (numerical stability)."""
        X = np.array([[1e15], [-1e15], [0.0]])
        scaler = StandardScaler()
        out = scaler.fit_transform(X)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_shape_mismatch(self):
        """Error due to mismatch in the number of test features"""
        X_train = np.random.rand(5, 3)
        X_test = np.random.rand(5, 2)
        scaler = StandardScaler()
        scaler.fit(X_train)
        with self.assertRaises(ValueError):
            scaler.transform(X_test)

if __name__ == '__main__':
    import HtmlTestRunner
    # Run all tests
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='reports'))