import numpy as np
from numcompute.utils import distances, sigmoid, activations,logsumexp,top_k,batching
import unittest

class UtilsTest(unittest.TestCase):
    def test_distances_wrong_shape(self):
        x1 = np.asarray([0.1, 0.3, 0.6, 0.8])
        x2 = np.asarray([0.2, 0.4])
        
        with self.assertRaises(ValueError):
            distances(x1, x2)
    
    def test_distances_wrong_method(self):
        x1 = np.asarray([0.1, 0.3, 0.6])
        x2 = np.asarray([0.2, 0.4, 0.7])
        
        with self.assertRaises(ValueError):
            distances(x1, x2, method='wrong method')
            
    def test_sigmoid_overflow(self):
        x = np.array([-1000, 0, 1000])
        result = np.array([0.0, 0.5, 1.0])
        
        self.assertTrue(np.allclose(sigmoid(x), result))
        
    def test_activations_softmax_overflow(self):
        x = np.asarray([-1000, 0, 1000])
        result = np.asarray([0.0, 0.0, 1.0])
        
        self.assertTrue(np.allclose(activations(x, method='softmax'), result))
    
    def test_activations_wrong_method(self):
        x = np.asarray([-1000, 0, 1000])
        
        with self.assertRaises(ValueError):
            activations(x, method='wrong method')
            
    def test_logsumexp_overflow(self):
        x = np.asarray([-1000, 0, 1000])
        result = 1000.0
        
        self.assertAlmostEqual(logsumexp(x), result)
        
    def test_top_k(self):
        x1 = np.asarray([0.8, 0.9, 0.7, 0.1])
        x2 = np.asarray([])
        k1 = 2
        k2 = 0
        k3 = 5
        
        with self.assertRaises(ValueError):
            top_k(x1, k2)
        
        with self.assertRaises(ValueError):
            top_k(x1, k3)
            
        with self.assertRaises(ValueError):
            top_k(x2, k1)
            
    def test_batching(self):
        X = np.asarray([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
        y = np.asarray([0, 1, 0, 1])
        y_wrong = np.asarray([0, 1, 0])
        batch_size = 2
        batch_size_wrong = 0
        
        with self.assertRaises(ValueError):
            batching(X, y_wrong, batch_size)
        
        with self.assertRaises(ValueError):
            batching(X, y, batch_size_wrong)
            
    if __name__ == '__main__':
        unittest.main()