import numpy as np

class StandardScaler:
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

        X_out = (X - self.mean_) / self.scale_
        self.scale_[self.scale_ == 0] = 1.0

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MinMaxScaler:
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
    def __init__(self, strategy="constant", fill_value=0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer_ = None
        self.n_features_in_ = None
        self._is_fitted = False

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
            raise ValueError("SimpleInputer has not been fitted yet.")
            # 2. 转数组
            # 3. 输入检查
            # 4. copy 一份
            # 5. 找到缺失值位置
            # 6. 替换成 fill_value
            # 7. 返回结果
        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)