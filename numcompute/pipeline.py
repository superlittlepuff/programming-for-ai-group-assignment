import numpy as np

class Pipeline:
    """Modular pipeline for chaining transformers and estimators."""
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X):
        """Fit all pipeline steps."""
        X_temp = X.copy()
        for name, step in self.steps:
            if hasattr(step, 'fit'):
                step.fit(X_temp)
            # Transform data for the next step if applicable
            if hasattr(step, 'transform'):
                X_temp = step.transform(X_temp)
        return self

    def transform(self, X):
        """Apply sequential transformations."""
        X_temp = X.copy()
        for name, step in self.steps:
            if hasattr(step, 'transform'):
                X_temp = step.transform(X_temp)
        return X_temp

    def fit_transform(self, X):
        """Fit and then transform data."""
        return self.fit(X).transform(X)
        
    def predict(self, X):
        """Transform data and predict using the final step."""
        X_temp = X.copy()
        
        # Pass data through all transformers except the last step
        for name, step in self.steps[:-1]:
            if hasattr(step, 'transform'):
                X_temp = step.transform(X_temp)
                
        # Predict with the final estimator
        last_step_name, last_model = self.steps[-1]
        
        if hasattr(last_model, 'predict'):
            return last_model.predict(X_temp)
        else:
            raise AttributeError(f"Final step '{last_step_name}' has no predict method.")