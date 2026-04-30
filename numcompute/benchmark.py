import numpy as np
import time
from numcompute.preprocessing import StandardScaler, SimpleImputer

def benchmark_scaler():
    print("Testing StandardScaler with 100000 rows")
    n_rows, n_cols = 100000, 10
    data = np.random.rand(n_rows, n_cols)
    
    # Python explicit loop implementation
    def loop_scaling(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        out = np.empty_like(X)
        for i in range(n_rows):
            for j in range(n_cols):
                out[i, j] = (X[i, j] - mean[j]) / (std[j] if std[j] != 0 else 1.0)
        return out

    # Test loop performance
    start_time = time.perf_counter()
    loop_scaling(data)
    loop_time = time.perf_counter() - start_time
    print(f"Python explicit loop time {loop_time:.4f} seconds")

    # Test NumPy vectorized implementation
    scaler = StandardScaler()
    start_time = time.perf_counter()
    scaler.fit_transform(data)
    vec_time = time.perf_counter() - start_time
    print(f"NumPy vectorized time {vec_time:.4f} seconds")
    
    print(f"Speedup {loop_time / vec_time:.2f} times")
    print("")
    return loop_time, vec_time

def benchmark_imputer():
    print("Testing SimpleImputer with 100000 rows")
    n_rows, n_cols = 100000, 10
    data = np.random.rand(n_rows, n_cols)
    # Inject 10 percent NaN values
    mask = np.random.choice([True, False], size=data.shape, p=[0.1, 0.9])
    data[mask] = np.nan

    # Python explicit loop implementation
    def loop_impute(X, fill_value=0.0):
        out = X.copy()
        for i in range(n_rows):
            for j in range(n_cols):
                if np.isnan(X[i, j]):
                    out[i, j] = fill_value
        return out

    start_time = time.perf_counter()
    loop_impute(data)
    loop_time = time.perf_counter() - start_time
    print(f"Python explicit loop time {loop_time:.4f} seconds")

    # Test NumPy vectorized implementation
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    start_time = time.perf_counter()
    imputer.fit_transform(data)
    vec_time = time.perf_counter() - start_time
    print(f"NumPy vectorized time {vec_time:.4f} seconds")

    print(f"Speedup {loop_time / vec_time:.2f} times")
    print("")
    return loop_time, vec_time

if __name__ == "__main__":
    print("Start performance benchmark")
    print("")
    s_loop, s_vec = benchmark_scaler()
    i_loop, i_vec = benchmark_imputer()
    
    print("Final benchmark summary report")
    print(f"StandardScaler speedup {s_loop/s_vec:.1f}x")
    print(f"SimpleImputer speedup {i_loop/i_vec:.1f}x")