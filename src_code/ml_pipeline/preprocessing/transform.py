# Use it in your pipeline like this:
from pathlib import Path
from typing import List
import joblib
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FunctionTransformer, Pipeline

from notebooks.constants import LINE_TOKEN_FEATURES, NUMERIC_FEATURES
from notebooks.logging_config import MyLogger
from notebooks.transformers import EmbeddingExpander, FeatureInteractionTransformer, NamingPCA, WinsorizerIQR
from src_code.config import FITTED_TRANSFORMER, SubsetType
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.preprocessing.vectorizers import sklearn_tfidf_vectorizer

set_config(transform_output="pandas")
log_transformer = FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one",)

PCA_CODE_EMB_COMPONENTS = 60
PCA_MSG_EMB_COMPONENTS = 80
WINSORIZE_FACTOR = 1.5
VARIANCE_THRESHOLD = 0.0

def build_transformer(random_state: int, logger: MyLogger = DEF_NOTEBOOK_LOGGER) -> ColumnTransformer:
    pipelines: List[Pipeline] = []

    code_emb_pipe = Pipeline(
        [
            ("expand", EmbeddingExpander(prefix="code")),
            (
                "pca",
                NamingPCA(
                    n_components=PCA_CODE_EMB_COMPONENTS, prefix="code_emb_", random_state=random_state
                ),
            ),
        ]
    )
    logger.log_result(f"PCA for code embeddings set to {PCA_CODE_EMB_COMPONENTS} components.", print_to_console=True)

    msg_emb_pipe = Pipeline(
        [
            ("expand", EmbeddingExpander(prefix="msg")),
            # ('pca', PCA(n_components=100, random_state=RANDOM_STATE))
            (
                "pca",
                NamingPCA(
                    n_components=PCA_MSG_EMB_COMPONENTS, prefix="msg_emb_", random_state=random_state
                ),
            ),
        ]
    )
    logger.log_result(f"PCA for message embeddings set to {PCA_MSG_EMB_COMPONENTS} components.", print_to_console=True)


    # 1. Define a pipeline for numeric features: Winsorize THEN Log
    numeric_pipe = Pipeline(
        [
            ("winsorize", WinsorizerIQR(factor=WINSORIZE_FACTOR)),
            ("log", log_transformer),
            # ("interactions", FeatureInteractionTransformer(NUMERIC_FEATURES)),
            ("var_thresh", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
        ]
    )
    logger.log_result(f"Winsorization factor set to {WINSORIZE_FACTOR}.", print_to_console=True)
    logger.log_result(f"Variance threshold set to {VARIANCE_THRESHOLD}.", print_to_console=True)

    pipelines.extend([msg_emb_pipe, code_emb_pipe, numeric_pipe])


    transformer = ColumnTransformer(
        transformers=[
            ("text", sklearn_tfidf_vectorizer, "message"),
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("tokens", log_transformer, LINE_TOKEN_FEATURES),
            ("code_embed", code_emb_pipe, ["code_embed"]),  # Pass as list
            ("msg_embed", msg_emb_pipe, ["msg_embed"]),  # Pass as list
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,  # This now works because names are unique
    )



    return transformer


def transform(
    df: pd.DataFrame,
    subset: SubsetType,
    random_state: int,
    logger: MyLogger = DEF_NOTEBOOK_LOGGER,
    fitted_transformer: Path = FITTED_TRANSFORMER,
    print_to_console: bool = True,
):
    logger.log_check("Performing df transformation...")
    # set_config(transform_output="pandas")
    # log_transformer = FunctionTransformer(np.log1p, validate=False)

    if subset == "train":
       
        logger.log_result(
            "Detected train subset. Creating new preprocessor...",
            print_to_console=print_to_console,
        )

        # 3. FIT and TRANSFORM
        transformer = build_transformer(random_state=random_state)

        transformer.fit(df)
        df = transformer.transform(df)

        feature_names = transformer.get_feature_names_out()


        # 4. SAVE
        joblib.dump(transformer, fitted_transformer)

        # print("Fitted preprocessor saved to fitted_preprocessor.joblib")
    elif subset in ("test", "val"):
        logger.log_result(
            "Detected test subset. Loading fitted preprocessor...",
            print_to_console=print_to_console,
        )
        transformer: ColumnTransformer = joblib.load(fitted_transformer)

        df = transformer.transform(df)
    else:
        msg = "Unknown subset value!"
        logger.logger.error(msg)
        raise ValueError(msg)

    logger.log_result(
        "Transformations applied successfully.", print_to_console=print_to_console
    )

    return df, transformer


def pca_explained_variance(transformer: ColumnTransformer, name: str) -> float:
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
    if not hasattr(transformer, "named_transformers_"):
        raise RuntimeError("Transformer must be fitted before accessing PCA info.")

    try:
        pca = transformer.named_transformers_[name].named_steps["pca"]
    except KeyError as e:
        raise KeyError(f"No PCA found under transformer '{name}'") from e

    return float(pca.explained_variance_ratio_.sum())
