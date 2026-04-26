import itertools
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.decomposition import PCA


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
                X_[col], a_min=self.lower_bounds[col], a_max=self.upper_bounds[col]
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


# class QuantileThresholdFlag(BaseEstimator, TransformerMixin):
#     """Learns a quantile threshold on fit and creates a boolean flag on transform."""

#     def __init__(self, quantile=0.95):
#         self.quantile = quantile
#         self.threshold = None

#     def fit(self, X, y=None):
#         # Assumes X is a DataFrame with a single column (the target feature)
#         self.threshold = X.iloc[:, 0].quantile(self.quantile)
#         return self

#     def transform(self, X):
#         # Apply the learned threshold
#         flag = (X.iloc[:, 0] > self.threshold).astype(int).values.reshape(-1, 1)
#         return flag

#     def get_feature_names_out(self, input_features=None):
#         # Renames the output feature
#         if input_features is not None and len(input_features) == 1:
#             return [f"extreme_flag_{input_features[0]}"]
#         return ["extreme_flag"]


class EmbeddingExpander(BaseEstimator, TransformerMixin):
    def __init__(self, prefix="emb"):
        self.prefix = prefix
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        # We just need to determine how many columns will be created
        first_val = np.array(X.iloc[0, 0])
        self.n_dims = first_val.shape[0]
        self.feature_names_out_ = [f"{self.prefix}_{i}" for i in range(self.n_dims)]
        return self

    def transform(self, X):
        # X will be a DataFrame with one column (e.g., 'code_embed')
        # where each row is a list/array
        col_data = X.iloc[:, 0].values
        expanded = np.vstack(col_data)

        return pd.DataFrame(expanded, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


# class FeatureInteractionTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, feature_names):
#         self.feature_names = feature_names  # store verbatim

#     def fit(self, X, y=None):
#         self._feature_names_ = list(self.feature_names)
#         self._interaction_names_ = [
#             f"{f1}_x_{f2}"
#             for i, f1 in enumerate(self._feature_names_)
#             for f2 in self._feature_names_[i + 1 :]
#         ]
#         return self

#     def transform(self, X):
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(
#                 X,
#                 columns=self._feature_names_,
#                 index=getattr(X, "index", None),
#             )

#         X = X.copy()

#         for f1, f2 in itertools.combinations(self._feature_names_, 2):
#             X[f"{f1}_x_{f2}"] = X[f1] * X[f2]

#         return X

#     def get_feature_names_out(self, input_features=None):
#         return np.array(self._feature_names_ + self._interaction_names_)


class NamingPCA(PCA):
    def __init__(self, n_components=100, prefix="pca", **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.prefix = prefix

    def get_feature_names_out(self, input_features=None):
        return [f"{self.prefix}{i}" for i in range(self.n_components)]


class ZeroHeavyFeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, max_zero_fraction: float = 0.95):
        self.max_zero_fraction = max_zero_fraction
        self.keep_columns_ = None
        self.drop_columns_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        zero_frac = (X == 0).mean(axis=0)
        self.keep_columns_ = zero_frac[zero_frac <= self.max_zero_fraction].index.tolist()
        self.drop_columns_ = zero_frac[zero_frac > self.max_zero_fraction].index.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X.loc[:, self.keep_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_columns_, dtype=object)


if __name__ == "__main__":
    pass
    