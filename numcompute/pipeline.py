import numpy as np

class Pipeline:
    """
    Chain multiple transformers into a single reusable workflow.

    Each step is expected to be a ``(name, transformer)`` pair. A transformer
    should provide ``fit`` and/or ``transform`` methods following the
    fit/transform API used across the NumCompute toolkit.

    Attributes
    ----------
    steps : list of tuple[str, object]
        Sequence of named transformer steps applied in order.
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
