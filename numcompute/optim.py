import numpy as np

def grad(f, x, h=1e-5, method='central'):
    if method not in ('central', 'forward'):
        raise ValueError("Gradient method should be central or forward")
    
    x = np.asarray(x,dtype=float)
    gradi = np.zeros(x.shape)
    
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
    x = np.asarray(x, dtype=float)
    
    Fx = np.array(F(x))
    if Fx.ndim == 0:
        raise ValueError("You should use grad for scalar")
    
    jaco = np.zeros((F(x).size,x.size))    
    for i in range(x.size):
        x1 = x.copy()
        x2 = x.copy()
        
        x1[i] += h
        x2[i] -= h
        
        jaco[:,i] = (F(x1) - F(x2)) / (2*h)
    
    return jaco


        