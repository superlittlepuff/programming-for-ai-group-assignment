import numpy as np
import unittest
from numcompute.optim import grad, jacobian

class OptimTest(unittest.TestCase):
    def test_grad_wrong_method(self):
        def f(x):
            return x**2
        
        x = np.asarray([0.1, 0.3, 0.6])
        
        with self.assertRaises(ValueError):
            grad(f, x, method='wrong method')
            
    def test_jacobian_wrong_scalar(self):
        # Test whether the jacobian function is a scalar function or not.
        # If it was a scalar function, it should raise a ValueError.
        def F(x):
            return x[0]**2 + x[1] + x[2]
        
        x = np.asarray([0.1, 0.3, 0.6])
        
        with self.assertRaises(ValueError):
            jacobian(F, x)
            
    if __name__ == '__main__':
        unittest.main()