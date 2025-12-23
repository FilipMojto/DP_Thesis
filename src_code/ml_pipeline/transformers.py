# Use it in your pipeline like this:
from typing import List
import numpy as np
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FunctionTransformer, Pipeline

from notebooks.constants import LINE_TOKEN_FEATURES, NUMERIC_FEATURES
from notebooks.transformers import EmbeddingExpander, NamingPCA, WinsorizerIQR


set_config(transform_output="pandas")
log_transformer = FunctionTransformer(np.log1p, validate=False)


class ReproducibleTransformer():

    def __init__(self, random_state: int, fitted_transformer: ColumnTransformer = None):
        self.pipelines: List[Pipeline] = []
        self.random_state = random_state
        # np.random.seed(random_state)
        
        if fitted_transformer is None:
            code_emb_pipe = Pipeline(
                [
                    ("expand", EmbeddingExpander(prefix="code")),
                    (
                        "pca",
                        NamingPCA(
                            n_components=10, prefix="code_emb_", random_state=self.random_state
                        ),
                    ),
                ]
            )

            msg_emb_pipe = Pipeline(
                [
                    ("expand", EmbeddingExpander(prefix="msg")),
                    # ('pca', PCA(n_components=100, random_state=RANDOM_STATE))
                    (
                        "pca",
                        NamingPCA(
                            n_components=45, prefix="msg_emb_", random_state=self.random_state
                        ),
                    ),
                ]
            )

            # 1. Define a pipeline for numeric features: Winsorize THEN Log
            numeric_pipe = Pipeline(
                [
                    ("winsorize", WinsorizerIQR(factor=1.5)),
                    ("log", log_transformer),
                    ("var_thresh", VarianceThreshold(threshold=0.0)),
                ]
            )

            self.pipelines.extend([msg_emb_pipe, code_emb_pipe, numeric_pipe])

            # embedding_transformer = Pipeline(steps=[
            #     ("pca", PCA(n_components=100, random_state=RANDOM_STATE))
            # ])

            # 2. Setup the ColumnTransformer
            # self.verb
            self.transformer = ColumnTransformer(
                transformers=[
                    # ('num_transformed', numeric_pipeline, NUMERIC_FEATURES),
                    # ('token_transformed', log_transformer, LINE_TOKEN_FEATURES),
                    # ("embed", embedding_transformer, FLATTENED_EMBEDDINGS),
                    ("num", numeric_pipe, NUMERIC_FEATURES),
                    ("tokens", log_transformer, LINE_TOKEN_FEATURES),
                    ("code_embed", code_emb_pipe, ["code_embed"]),  # Pass as list
                    ("msg_embed", msg_emb_pipe, ["msg_embed"]),  # Pass as list
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,  # This now works because names are unique
            )
        else:
            self.transformer = fitted_transformer
    
    def pca_explained_variance(self, name: str) -> float:
        """
        Return total explained variance ratio for a PCA step
        inside a named ColumnTransformer sub-pipeline.

        Parameters
        ----------
        name : str
            Name of the transformer in ColumnTransformer
            (e.g. 'code_embed', 'msg_embed')

        Returns
        -------
        float
            Sum of explained variance ratios
        """
        if not hasattr(self.transformer, "named_transformers_"):
            raise RuntimeError("Transformer must be fitted before accessing PCA info.")

        try:
            pca = (
                self.transformer
                .named_transformers_[name]
                .named_steps["pca"]
            )
        except KeyError as e:
            raise KeyError(f"No PCA found under transformer '{name}'") from e

        return float(pca.explained_variance_ratio_.sum())
