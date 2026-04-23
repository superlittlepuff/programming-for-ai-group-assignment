import numpy as np

def distances(x1, x2, method="euclidean"):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    if x1.shape != x2.shape:
        raise ValueError("The shape of two arrays are not the same")
    
    if method not in ("euclidean", "manhattan"):
        raise ValueError("Method should be euclidean or manhattan")
    
    if method == "euclidean":
        return np.sqrt(np.sum((x1 - x2)**2))
    elif method == "manhattan":
        return np.sum(np.abs(x1 - x2))

def activations(x, method="relu"):
    x = np.asarray(x)
    
    if method not in ("relu", "sigmoid", "tanh", "softmax"):
        raise ValueError("Method should be relu, sigmoid, tanh, or softmax")

    if method == "relu":
        return np.maximum(0, x)
    elif method == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif method == "tanh":
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    elif method == "softmax":
        return np.exp(x) / np.sum(np.exp(x))
    
def logsumexp(x):
    x = np.asarray(x)
    return np.log(np.sum(np.exp(x)))

def top_k(x, k):
    x = np.asarray(x)
    
    if x.size == 0:
        raise ValueError("The input array is empty")
    
    if k <= 0:
        raise ValueError("k should be a positive number")
    
    if k > x.size:
        raise ValueError("k exceeds the length of the array")
    
    return np.argsort(x)[::-1][:k]

def batching(X, y, batch_size):
    if batch_size <= 0:
        raise ValueError("Batch size should be a positive number")
    
    if len(X) != len(y):
        raise ValueError("The length of X and y should be the same")
    
    batch_result = []
    for i in range(0, len(X), batch_size):
        batch_result.append((X[i:i+batch_size], y[i:i+batch_size]))
    return batch_result