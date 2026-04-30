import numpy as np

class Pipeline:
    """
    Modular pipeline to chain multiple transformers.
    Supports fit, transform, and fit_transform semantics.
    """
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X):
        X_temp = X.copy()
        for name, transformer in self.steps:
            if hasattr(transformer, 'fit'):
                transformer.fit(X_temp)
            if hasattr(transformer, 'transform'):
                X_temp = transformer.transform(X_temp)
        return self

    def transform(self, X):
        X_temp = X.copy()
        for name, transformer in self.steps:
            if hasattr(transformer, 'transform'):
                X_temp = transformer.transform(X_temp)
        return X_temp

    def fit_transform(self, X):
        return self.fit(X).transform(X)