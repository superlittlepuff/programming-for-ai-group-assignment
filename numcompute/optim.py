import numpy as np

def grad(f, x, h=1e-5, method='central'):
    """
    Compute the numerical gradient of the function f at a specified element using different methods.

    Args:
        f (fuction): The function for which the gradient is to be computed.
        x (array): The array at which the gradient is to be computed.
        h (_type_, optional): The magic number which pervents underflow. Defaults to 1e-5.
        method (str, optional): The method to compute the gradient. Defaults to 'central'.

    Raises:
        ValueError: if the method is not central or forward.

    Returns:
        np.array: the numerical gradient of the function f at a specified element using different methods.
    """
    
    if method not in ('central', 'forward'):
        raise ValueError("Gradient method should be central or forward")
    
    x = np.asarray(x,dtype=float)
    gradi = np.zeros(x.shape, dtype=float)
    
    for i,_ in enumerate(gradi):
        x1 = x.copy()
        x1[i] += h
        if method == 'central':
            x2 = x.copy()
            x2[i] -= h
            gradi[i] = (f(x1) - f(x2)) / (2*h)
        elif method == 'forward':
            gradi[i] = (f(x1) - f(x)) / h
    
    return gradi

def jacobian(F, x, h=1e-5):
    """
    Compute the numerical Jacobian of the function F at a specified element using central difference method.

    Args:
        F (Functions): Functions for which the Jacobian is to be computed.
        x (_type_): The array at which the Jacobian is to be computed.
        h (_type_, optional): The magic number. Defaults to 1e-5.

    Raises:
        ValueError: if the function F is scalar.

    Returns:
        np.array:the numerical Jacobian of the function F at a specified element using central difference method.
    """
    
    x = np.asarray(x, dtype=float)
    
    Fx = np.array(F(x))
    if Fx.ndim == 0:
        raise ValueError("You should use grad for scalar")
    
    jaco = np.zeros((F(x).size,x.size), dtype=float)    
    for i in range(x.size):
        x1 = x.copy()
        x2 = x.copy()
        
        x1[i] += h
        x2[i] -= h
        
        jaco[:,i] = (F(x1) - F(x2)) / (2*h)
    
    return jaco


        