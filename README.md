# NumCompute Toolkit


NumCompute is a high-performance, modular scientific computing toolkit built entirely on plain Python and NumPy. It provides reusable, vectorized components for data preprocessing, statistical analysis, mathematical optimization, and performance evaluation, completely avoiding slow Python loops in core computations.

##  Installation

This toolkit requires no external Machine Learning libraries (e.g., scikit-learn, pandas). The only required dependency is `numpy`.

**Option A: Run from the submitted ZIP archive (Recommended for assessment)**
Since the complete project is submitted as a `.zip` archive containing the local `.git` history, please follow these steps:
# ```bash
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

## 4. Team Members
- Zhiyang Ma (a1975426)
- Tianyu Zhang (a1944038)
- Guilin Luo (a1989840)
- Dan Tran (a3188145)

