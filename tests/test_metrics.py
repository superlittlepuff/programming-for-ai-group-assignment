import unittest
import numpy as np
from numcompute.metrics import accuracy,precision,recall,f1,confusion_matrix,mse,roc_curve

class MetricsTest(unittest.TestCase):
    def test_distance_wrong_shape(self):
        y_true = np.asarray([0, 1, 0, 1, 0])
        y_pred = np.asarray([0, 1, 0])
        
        with self.assertRaises(ValueError):
            accuracy(y_true,y_pred)
    
    def test_precision_wrong_shape(self):
        y_true = np.asarray([0, 1, 0, 1, 0])
        y_pred = np.asarray([0, 1, 0])
        
        with self.assertRaises(ValueError):
            precision(y_true, y_pred)
            
    def test_precision_divide_zero(self):
        # TP + FP = 0
        y_true = np.asarray([1, 0, 1, 0])
        y_pred = np.asarray([0, 0, 0, 0])

        self.assertEqual(0.0,precision(y_true, y_pred))
    
    def test_recall_wrong_shape(self):
        y_true = np.asarray([0, 1, 0, 1, 0])
        y_pred = np.asarray([0, 1, 0])
        
        with self.assertRaises(ValueError):
            recall(y_true, y_pred)
    
    def test_recall_divide_zero(self):
        # TP + FN = 0
        y_true = np.asarray([0, 0, 0, 0, 0])
        y_pred = np.asarray([1, 0, 1, 0, 1])

        self.assertEqual(0.0,recall(y_true, y_pred))
        
    def test_f1_wrong_shape(self):
        y_true = np.asarray([0, 1, 0, 1, 0])
        y_pred = np.asarray([0, 1, 0])
        
        with self.assertRaises(ValueError):
            f1(y_true, y_pred)
    
    def test_f1_divide_zero(self):
        # precision = 0 and recall = 0
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1])

        self.assertEqual(0.0, f1(y_true, y_pred))
        
    def test_confusion_matrix_wrong_shape(self):
        y_true = np.asarray([0, 1, 0, 1, 0])
        y_pred = np.asarray([0, 1, 0])
        
        with self.assertRaises(ValueError):
            confusion_matrix(y_true, y_pred)
            
    def test_mse_wrong_shape(self):
        y_true = np.asarray([0, 1, 0, 1, 0])
        y_pred = np.asarray([0, 1, 0])
        
        with self.assertRaises(ValueError):
            mse(y_true, y_pred)
    
    def test_roc_curve_wrong_shape(self):
        y_true = np.asarray([0, 1, 0, 1, 0])
        y_scores = np.asarray([0.8, 0.6, 0.2])
        
        with self.assertRaises(ValueError):
            roc_curve(y_true, y_scores)
    
    if __name__ == '__main__':    
        unittest.main()
    