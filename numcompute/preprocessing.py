import numpy as np

class StandardScaler:
    """
    Standardize numerical features using z-score normalization.

    This transformer computes the mean and standard deviation of each
    feature during ``fit`` and applies feature-wise standardization during
    ``transform``.

    Attributes
    ----------
    mean_ : np.ndarray or None
        Per-feature mean learned during fitting.
    scale_ : np.ndarray or None
        Per-feature standard deviation learned during fitting.
    n_features_in_ : int or None
        Number of input features seen during fitting.
    _is_fitted : bool
        Whether the transformer has been fitted.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self._is_fitted = False

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("The dimension of X should be 2")

        if X.size == 0:
            raise ValueError("The input array is empty")

        self.n_features_in_ = X.shape[1]

        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)

        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise ValueError("StandardScaler has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("The dimension of X should be 2")

        if X.size == 0:
            raise ValueError("The input array is empty")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count does not match fitted data.")

        self.scale_[self.scale_ == 0] = 1.0
        X_out = (X - self.mean_) / self.scale_

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MinMaxScaler:
    """
    Scale numerical features to the [0, 1] range.

    This transformer computes the minimum and maximum value of each feature
    during ``fit`` and applies min-max scaling during ``transform``.

    Attributes
    ----------
    max_ : np.ndarray or None
        Per-feature maximum values.
    min_ : np.ndarray or None
        Per-feature minimum values.
    range_ : np.ndarray or None
        Per-feature ranges (max - min), adjusted to avoid division by zero.
    n_features_in_ : int or None
        Number of input features seen during fitting.
    _is_fitted : bool
        Whether the transformer has been fitted.
    """
    def __init__(self):
        self.max_ = None
        self.min_ = None
        self.range_ = None
        self.n_features_in_ = None
        self._is_fitted = False

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("The dimension of X should be 2")
        if X.size == 0:
            raise ValueError("The input array is empty")

        self.n_features_in_ = X.shape[1]

        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0

        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise ValueError("MinMaxScaler has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("The dimension of X should be 2")

        if X.size == 0:
            raise ValueError("The input array is empty")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count does not match fitted data.")

        X_out = (X - self.min_) / self.range_

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class OneHotEncoder:
    """
    Encode categorical features as one-hot numeric arrays.

    During ``fit``, the encoder stores the unique categories observed in
    each categorical feature column. During ``transform``, each category is
    mapped to a binary indicator vector.

    Attributes
    ----------
    categories_ : list of np.ndarray or None
        Unique categories for each input feature column.
    n_features_in_ : int or None
        Number of categorical feature columns seen during fitting.
    _is_fitted : bool
        Whether the encoder has been fitted.
    """
    def __init__(self):
        self.categories_ = None
        self.n_features_in_ = None
        self._is_fitted = False

    def fit(self, X):
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("The dimension of X should be 2")
        if X.size == 0:
            raise ValueError("The input array is empty")

        self.n_features_in_ = X.shape[1]
        self.categories_ = []

        for i in range(self.n_features_in_):
            col = X[:, i]
            cats = np.unique(col)
            self.categories_.append(cats)

        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise ValueError("OneHotEncoder has not been fitted yet.")

        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("The dimension of X should be 2")
        if X.size == 0:
            raise ValueError("The input array is empty")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count does not match fitted data.")

        encoded_columns = []

        for i in range(self.n_features_in_):
            col = X[:, i]
            cats = self.categories_[i]
            one_hot = np.zeros((len(col), len(cats)))

            for row_idx in range(len(col)):
                value = col[row_idx]

                if value not in cats:
                    raise ValueError(f"Unknown category '{value}' in column {i}")

                cat_idx = np.where(cats == value)[0][0]
                one_hot[row_idx, cat_idx] = 1

            encoded_columns.append(one_hot)

        X_out = np.concatenate(encoded_columns, axis=1)
        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class SimpleImputer:
    """
    Replace missing values with a constant fill value.

    The current implementation supports constant-value imputation for
    numeric arrays containing ``np.nan``.

    Attributes
    ----------
    strategy : str
        Imputation strategy. Currently intended for constant filling.
    fill_value : float or int
        Constant value used to replace missing entries.
    n_features_in_ : int or None
        Number of input features seen during fitting.
    _is_fitted : bool
        Whether the imputer has been fitted.
    statistics_ : scalar or None
        Stored fill value used during transformation.
    """
    def __init__(self, strategy="constant", fill_value=0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.n_features_in_ = None
        self._is_fitted = False
        self.statistics_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("The dimension of X should be 2")
        if X.size == 0:
            raise ValueError("The input array is empty")

        self.n_features_in_ = X.shape[1]

        self.statistics_ = self.fill_value
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise ValueError("SimpleImputer has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.size == 0:
            raise ValueError("Input array is empty.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count does not match fitted data.")

        X_out = X.copy()

        mask = np.isnan(X_out)
        X_out[mask] = self.statistics_

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)