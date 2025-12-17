import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.compose import ColumnTransformer

from src_code.config import SubsetType

class WinsorizerIQR(BaseEstimator, TransformerMixin):
    """
    Custom transformer to cap outliers using IQR method.
    Calculates bounds on fit, applies them on transform.
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        # Learn the bounds ONLY from the training data (X)
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds[col] = Q1 - self.factor * IQR
            self.upper_bounds[col] = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        # Apply the learned bounds to the input data
        X_ = X.copy()
        for col in X_.columns:
            X_[col] = np.clip(
                X_[col],
                a_min=self.lower_bounds[col],
                a_max=self.upper_bounds[col]
            )
        return X_
    
    # NEW METHOD REQUIRED BY COLUMNTRANSFORMER
    def get_feature_names_out(self, input_features=None):
        """Returns the names of the output features."""
        # For a simple column-wise transformation, the output names are the input names.
        if input_features is None:
            # If input_features is not passed (older Sklearn versions), 
            # assume it's the keys learned during fit, if available.
            return list(self.lower_bounds.keys())
        
        # Scikit-learn passes the names of the columns this transformer operates on
        return input_features
    

class QuantileThresholdFlag(BaseEstimator, TransformerMixin):
    """Learns a quantile threshold on fit and creates a boolean flag on transform."""
    def __init__(self, quantile=0.95):
        self.quantile = quantile
        self.threshold = None

    def fit(self, X, y=None):
        # Assumes X is a DataFrame with a single column (the target feature)
        self.threshold = X.iloc[:, 0].quantile(self.quantile)
        return self

    def transform(self, X):
        # Apply the learned threshold
        flag = (X.iloc[:, 0] > self.threshold).astype(int).values.reshape(-1, 1)
        return flag

    def get_feature_names_out(self, input_features=None):
        # Renames the output feature
        if input_features is not None and len(input_features) == 1:
            return [f"extreme_flag_{input_features[0]}"]
        return ["extreme_flag"]
    

# def apply_transformer(subset: SubsetType):
#     msg = None

#     if subset == 'train':
#         # log_check("Detected train subset. Creating new preprocessor...", print_to_console=True)
#         msg = "Detected train subset. Creating new preprocessor..."
#         preprocessor = ColumnTransformer(
#             # ... existing transformers ...
#             ('extreme_flags', QuantileThresholdFlag(quantile=0.95), EXTREME_FLAG_FEATURES),
#             # ...
#         )

#         preprocessor.fit(df)
#         df = preprocessor.transform(df)
#         joblib.dump(preprocessor, ENGINEERING_PREPROCESSOR)
#     elif subset in ('test', 'validate'):
#         # log_check("Detected test subset. Loading fitted preprocessor...", print_to_console=True)
#         msg = "Detected test subset. Loading fitted preprocessor..."
#         loaded_preprocessor = joblib.load(ENGINEERING_PREPROCESSOR)
#         df = loaded_preprocessor.transform(df)
#     else:
#         msg = "Unknown subset value!"
#         # logger.error(msg)
#         # raise ValueError(msg)

#     return msg