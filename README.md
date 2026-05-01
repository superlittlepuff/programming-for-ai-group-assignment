# NumCompute Toolkit


NumCompute is a high-performance, modular scientific computing toolkit built entirely on plain Python and NumPy. It provides reusable, vectorized components for data preprocessing, statistical analysis, mathematical optimization, and performance evaluation, completely avoiding slow Python loops in core computations.

##  Installation

This toolkit requires no external Machine Learning libraries (e.g., scikit-learn, pandas). The only required dependency is `numpy`.

**Option A: Run from the submitted ZIP archive (Recommended for assessment)**
Since the complete project is submitted as a `.zip` archive containing the local `.git` history, please follow these steps:
# ```bash
programming-for-ai-group-assignment/
├── numcompute/
├── tests/
├── reports/
├── README.md
└── pyproject.toml
# 1. Unzip the submitted file
unzip <programming-for-ai-group-assignment>.zip
cd <programming-for-ai-group-assignment>

# 2. Install NumPy
pip install numpy

# **Option B: Clone from the repository
git clone [https://github.com/superlittlepuff/programming-for-ai-group-assignment.git](https://github.com/superlittlepuff/programming-for-ai-group-assignment.git)
cd programming-for-ai-group-assignment
pip install numpy

## 1. Data I/O and Preprocessing
This section implements the data loading and preprocessing components of the toolkit.

### io.py
The io.py module handles the transition of raw data into NumPy arrays.
- Features: Load CSV files via numpy.genfromtxt, custom delimiters, header skipping, and missing value handling.

### preprocessing.py
This module provides a suite of reusable classes following the fit/transform API protocol.
- Components: StandardScaler, MinMaxScaler, OneHotEncoder, and SimpleImputer.

---

## 2. Metrics, Optimization, and Utility
This section focuses on numerical computing methods for model evaluation and mathematical stability.

### metrics.py
- Indicators: Accuracy, Precision, Recall, F1-score, Confusion Matrix, MSE, and ROC-AUC.
- Robustness: Handles edge cases like zero denominators in metric calculations.

### optim.py
- Functions: Implements numerical differentiation (grad) using central difference by default, and Jacobian matrix estimation.

### utils.py
- Stability: Explicit overflow handling for Sigmoid, Softmax, and Logsumexp.
- Tools: Distance computations, top-k selection, and batching logic.

---

## 3. Testing and Edge Cases
The toolkit is verified through 33 automated unit tests to ensure exceptional robustness.
- Validation: Input shape checks, divide-by-zero protection, overflow defense, and API consistency checks.

---

## 4. Pipeline

The `Pipeline` module provides a simple way to chain multiple transformers into a single workflow. Each step follows the shared API used in this toolkit, allowing the output of one step to be passed directly into the next one.

The pipeline supports two kinds of components:

- **Preprocessors**: `fit`, `transform`
- **Models**: `fit`, `predict` *(no ML model implementation required)*

For example, a pipeline can first fill missing values with `SimpleImputer`, then normalize numeric features with `StandardScaler`, and finally return the transformed result in a single reusable object. If the final step is a model-like component, the same pipeline structure can also support `fit(X, y)` and `predict(X)`.

This design reduces manual preprocessing errors in several ways:
- it enforces a consistent order of operations
- it avoids repeatedly handling intermediate arrays by hand
- it keeps preprocessing and prediction logic in one reusable workflow
- it helps maintain consistency between training-time and inference-time transformations

As a result, the workflow becomes easier to reuse, easier to debug, and more consistent across experiments.

### Example: preprocessing-only pipeline

#```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import SimpleImputer, StandardScaler

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", StandardScaler())
])

X_tr = pipe.fit_transform(X)

---

## 5. Example Usage

#```python
import numpy as np
from numcompute.io import load_csv
from numcompute.preprocessing import SimpleImputer, StandardScaler
from numcompute.pipeline import Pipeline

# Load raw data
X = load_csv("student_performance_data.csv", delimiter=",", skip=1)

# Example numeric subset for preprocessing
X_numeric = np.array([
    [20.0, np.nan, 88.0],
    [21.0, 4.0, 91.0],
    [19.0, 2.8, np.nan]
])

# Build a preprocessing pipeline
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", StandardScaler())
])

# Fit and transform the data
X_processed = pipe.fit_transform(X_numeric)

print("Original numeric data:")
print(X_numeric)

print("Processed numeric data:")
print(X_processed)

---

## 6. Team Members
- Zhiyang Ma (a1975426)
- Tianyu Zhang (a1944038)
- Guilin Luo (a1989840)
- Dan Tran (a3188145)

