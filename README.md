Data I/O and Preprocessing

This section of the project implements the data loading and preprocessing components of the `NumCompute` toolkit.

`io.py`
The `io.py` module loads CSV files into NumPy arrays.

Features:
- load CSV files using `numpy.genfromtxt`
- support custom delimiters such as comma and tab
- skip header rows
- handle missing values
- return data as NumPy arrays
- basic error handling for missing or empty files

Example:
python
from numcompute.io import load_csv

X = load_csv("data/student_performance_demo.csv", delimiter=",", skip=1)
print(X)
preprocessing.py

The preprocessing.py module provides several reusable preprocessing classes with a consistent API:

fit(X) -> self
transform(X) -> X_out
fit_transform(X) -> X_out

Implemented classes:

StandardScaler

Applies z-score standardization to numerical features.

MinMaxScaler

Scales numerical features to the range [0, 1].

OneHotEncoder

Encodes categorical features into one-hot vectors.

SimpleImputer

Fills missing values using a constant value.

Example:

from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer

imputer = SimpleImputer(strategy="constant", fill_value=0)
X_filled = imputer.fit_transform(X_numeric)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)
