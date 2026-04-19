import numpy as np

def distances(x1, x2, method="euclidean"):
    if method == "euclidean":
        return np.sqrt(np.sum((x1 - x2)**2))
    elif method == "manhattan":
        return np.sum(np.abs(x1 - x2))

def activations(x, method="relu"):
    if method == "relu":
        return np.maximum(0, x)
    elif method == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif method == "tanh":
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    elif method == "softmax":
        return np.exp(x) / np.sum(np.exp(x))
    
def logsumexp(x):
    return np.log(np.sum(np.exp(x)))

# top-k helpers, batching
def top_k(x, k):
    return np.argsort(x)[:k]

def batching(X, y, batch_size):
    batch_result = []
    for i in range(0, len(X), batch_size):
        batch_result.append((X[i:i+batch_size], y[i:i+batch_size]))
    return batch_result