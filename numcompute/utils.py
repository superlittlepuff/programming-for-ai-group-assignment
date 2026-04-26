import numpy as np

def distances(x1, x2, method="euclidean")->float:
    """
    Compute the distance between two arrays.

    Args:
        x1 (array-like input): The first input array.
        x2 (array-like input): The second input array.
        method (str, optional): 
            Distance method either euclidean or manhattan. Defaults to "euclidean".

    Raises:
        ValueError: if the shape of two arrays are not the same
        ValueError: if method is not euclidean or manhattan

    Returns:
        float: The distance between two arrays.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    
    if x1.shape != x2.shape:
        raise ValueError("The shape of two arrays are not the same")
    
    if method not in ("euclidean", "manhattan"):
        raise ValueError("Method should be euclidean or manhattan")
    
    if method == "euclidean":
        return np.sqrt(np.sum((x1 - x2)**2))
    elif method == "manhattan":
        return np.sum(np.abs(x1 - x2))

def sigmoid(x):
    """
    Activation function that maps any numpy array to a value between 0 and 1.
    
    Args:
        x (array-like input): The input array.

    Returns:
        np.array: The output array after applying the sigmoid function, 
        which has prevented overflow. 
    """
    x = np.asarray(x, dtype=float)
    pos_x_idx = x >= 0
    neg_x_idx = x < 0
    result = np.zeros_like(x, dtype=float)
    
    result[pos_x_idx] = 1 / (1 + np.exp(-x[pos_x_idx]))
    
    exp_x = np.exp(x[neg_x_idx])
    result[neg_x_idx] = exp_x / (1 + exp_x) 
    return result

def activations(x, method="relu", axis = None)->np.array:
    """
    Apply the activation function to the input array.

    Args:
        x (array-like input): The input array.
        method (str, optional): 
        Activation method among relu,sigmoid,tanh,softmax. Defaults to "relu".
        axis (int, optional): The axis along which the softmax function is applied. Defaults to None.

    Raises:
        ValueError: if the method is not relu, sigmoid, tanh, or softmax

    Returns:
        np.array: The output array that has been computed by the activation function.
    """
    
    x = np.asarray(x, dtype=float)
    
    if method not in ("relu", "sigmoid", "tanh", "softmax"):
        raise ValueError("Method should be relu, sigmoid, tanh, or softmax")

    if method == "relu":
        return np.maximum(0, x)
    elif method == "sigmoid":
        return sigmoid(x)
    elif method == "tanh":
        return np.tanh(x)
    elif method == "softmax":
        max_x = np.max(x, axis=axis, keepdims=True)
        x = x - max_x # To prevent the overflow
        return np.exp(x) / np.sum(np.exp(x, axis=axis, keepdims=True))
    
def logsumexp(x, axis=None)->float:
    """
    Compute the log sum exponential of the input array.

    Args:
        x (array-like input): The input array.
        axis (int, optional): The axis along which the log sum exponential is computed. Defaults to None.

    Returns:
        float: The log sum exponential of the input array, 
        which has prevented overflow.
    """
    
    x = np.asarray(x, dtype=float)
    c = np.max(x, axis=axis, keepdims=True)
    return c + np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True)) # To prevent the overflow

def top_k(x, k)->np.array:
    """
    Get the top k indices of the input array.

    Args:
        x (array-like input): The input array.
        k (int number): The number of top k.

    Raises:
        ValueError: The input array is empty
        ValueError: k is a negative number or zero
        ValueError: k exceeds the length of the array

    Returns:
        np.array: The top k indices of the input array.
    """
    
    x = np.asarray(x,dype=float)
    
    if x.size == 0:
        raise ValueError("The input array is empty")
    
    if k <= 0:
        raise ValueError("k should be a positive number")
    
    if k > x.size:
        raise ValueError("k exceeds the length of the array")
    
    return np.argsort(x)[::-1][:k]

def batching(X, y, batch_size)->list:
    """
    Group elements of the input array into batches.

    Args:
        X (array-like input): The input array of features.
        y (array-like input): The input array of labels.
        batch_size (int number): The number of samples in each batch.

    Raises:
        ValueError: if batch_size is a negative number or zero
        ValueError: if the number of samples in X and y are not the same

    Returns:
        List of tuples: Every tuple contains a batch_size number of samples of X and y.
    """
    
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    
    if batch_size <= 0:
        raise ValueError("Batch size should be a positive number")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y should be the same")
    
    batch_result = []
    for i in range(0, X.shape[0], batch_size):
        batch_result.append((X[i:i+batch_size], y[i:i+batch_size]))
        
    return batch_result