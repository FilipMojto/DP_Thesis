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

from notebooks.constants import BINARY_BUCKET_FEATURES, ENGINEERED_FEATURES, HEAVY_TAIL_FEATURES, LINE_TOKEN_FEATURES, NUMERIC_FEATURES, SPARSE_TOKEN_FEATURES
from notebooks.logging_config import MyLogger
from notebooks.transformers import EmbeddingExpander, NamingPCA, WinsorizerIQR, ZeroHeavyFeatureDropper
from src_code.config import FITTED_TRANSFORMER, SubsetType
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.preprocessing.vectorizers import sklearn_tfidf_vectorizer

# set_config(transform_output="pandas")
set_config(transform_output="default")

log_transformer = FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one",)

PCA_CODE_EMB_COMPONENTS = 60
PCA_MSG_EMB_COMPONENTS = 80
WINSORIZE_FACTOR = 1.5
VARIANCE_THRESHOLD = 0.0

# def build_transformer(random_state: int, logger: MyLogger = DEF_NOTEBOOK_LOGGER) -> ColumnTransformer:
#     # pipelines: List[Pipeline] = []

#     code_emb_pipe = Pipeline(
#         [
#             ("expand", EmbeddingExpander(prefix="code")),
#             (
#                 "pca",
#                 NamingPCA(
#                     n_components=PCA_CODE_EMB_COMPONENTS, prefix="code_emb_", random_state=random_state
#                 ),
#             ),
#         ]
#     )
#     logger.log_result(f"PCA for code embeddings set to {PCA_CODE_EMB_COMPONENTS} components.", print_to_console=True)

#     msg_emb_pipe = Pipeline(
#         [
#             ("expand", EmbeddingExpander(prefix="msg")),
#             # ('pca', PCA(n_components=100, random_state=RANDOM_STATE))
#             (
#                 "pca",
#                 NamingPCA(
#                     n_components=PCA_MSG_EMB_COMPONENTS, prefix="msg_emb_", random_state=random_state
#                 ),
#             ),
#         ]
#     )
#     logger.log_result(f"PCA for message embeddings set to {PCA_MSG_EMB_COMPONENTS} components.", print_to_console=True)


#     heavy_tail_pipe = Pipeline([
#         ("winsorize", WinsorizerIQR(factor=WINSORIZE_FACTOR)),
#         ("log", log_transformer),
#         ("var_thresh", VarianceThreshold(threshold=0.0)),
#     ])

#     binary_pipe = Pipeline([
#         ("var_thresh", VarianceThreshold(threshold=0.0)),
#     ])

#     sparse_pipe = Pipeline([
#         ("drop_zero_heavy", ZeroHeavyFeatureDropper(max_zero_fraction=0.95)),
#         ("log", log_transformer),
#         ("var_thresh", VarianceThreshold(threshold=0.0)),
#     ])

#     transformer = ColumnTransformer(
#         transformers=[
#             ("text", sklearn_tfidf_vectorizer, "message"),
#             ("heavy_num", heavy_tail_pipe, HEAVY_TAIL_FEATURES),
#             ("binary", binary_pipe, BINARY_BUCKET_FEATURES),
#             ("sparse_tokens", sparse_pipe, SPARSE_TOKEN_FEATURES),
#             ("code_embed", code_emb_pipe, ["code_embed"]),
#             ("msg_embed", msg_emb_pipe, ["msg_embed"]),
#         ],
#         remainder="drop",
#         verbose_feature_names_out=False,
#     )

#     return transformer

def build_transformer(
    available_columns: list[str],
    random_state: int,
    logger: MyLogger = DEF_NOTEBOOK_LOGGER,
) -> ColumnTransformer:
    available_columns = set(available_columns)

    code_emb_pipe = Pipeline(
        [
            ("expand", EmbeddingExpander(prefix="code")),
            (
                "pca",
                NamingPCA(
                    n_components=PCA_CODE_EMB_COMPONENTS,
                    prefix="code_emb_",
                    random_state=random_state,
                ),
            ),
        ]
    )
    logger.log_result(
        f"PCA for code embeddings set to {PCA_CODE_EMB_COMPONENTS} components.",
        print_to_console=True,
    )

    msg_emb_pipe = Pipeline(
        [
            ("expand", EmbeddingExpander(prefix="msg")),
            (
                "pca",
                NamingPCA(
                    n_components=PCA_MSG_EMB_COMPONENTS,
                    prefix="msg_emb_",
                    random_state=random_state,
                ),
            ),
        ]
    )
    logger.log_result(
        f"PCA for message embeddings set to {PCA_MSG_EMB_COMPONENTS} components.",
        print_to_console=True,
    )

    heavy_tail_pipe = Pipeline([
        ("winsorize", WinsorizerIQR(factor=WINSORIZE_FACTOR)),
        ("log", log_transformer),
        ("var_thresh", VarianceThreshold(threshold=0.0)),
    ])

    binary_pipe = Pipeline([
        ("var_thresh", VarianceThreshold(threshold=0.0)),
    ])

    sparse_pipe = Pipeline([
        ("drop_zero_heavy", ZeroHeavyFeatureDropper(max_zero_fraction=0.95)),
        ("log", log_transformer),
        ("var_thresh", VarianceThreshold(threshold=0.0)),
    ])

    transformers = []

    # text
    if "message" in available_columns:
        transformers.append(("text", sklearn_tfidf_vectorizer, "message"))
        logger.log_result("Using text feature: message")

    # heavy-tail numeric
    heavy_cols = [c for c in HEAVY_TAIL_FEATURES if c in available_columns]
    if heavy_cols:
        transformers.append(("heavy_num", heavy_tail_pipe, heavy_cols))
        logger.log_result(f"Using heavy-tail features ({len(heavy_cols)}): {heavy_cols}")

    # binary / bucket
    binary_cols = [c for c in BINARY_BUCKET_FEATURES if c in available_columns]
    if binary_cols:
        transformers.append(("binary", binary_pipe, binary_cols))
        logger.log_result(f"Using binary features ({len(binary_cols)}): {binary_cols}")

    # sparse token features
    sparse_cols = [c for c in SPARSE_TOKEN_FEATURES if c in available_columns]
    if sparse_cols:
        transformers.append(("sparse_tokens", sparse_pipe, sparse_cols))
        logger.log_result(f"Using sparse token features ({len(sparse_cols)}): {sparse_cols}")

    # embeddings
    if "code_embed" in available_columns:
        transformers.append(("code_embed", code_emb_pipe, ["code_embed"]))
        logger.log_result("Using code embeddings")

    if "msg_embed" in available_columns:
        transformers.append(("msg_embed", msg_emb_pipe, ["msg_embed"]))
        logger.log_result("Using message embeddings")

    if not transformers:
        raise ValueError("No valid transformers could be constructed from available columns.")

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return transformer

def log_dropped_features(transformer, numeric_features_list, logger: MyLogger):
    # 1. Access the 'num' pipeline from the ColumnTransformer
    num_pipe = transformer.named_transformers_['num']
    
    # 2. Access the VarianceThreshold step (it's the last step in your pipe)
    selector = num_pipe.named_steps['var_thresh']
    
    # 3. Get the mask (True = kept, False = dropped)
    # Note: If you have steps before var_thresh that change feature count, 
    # you might need to get names from the step immediately preceding it.
    mask = selector.get_support()
    
    # Names entering the VarianceThreshold are those after winsorizing/logging
    # In your case, it's NUMERIC_FEATURES + ENGINEERED_FEATURES
    input_names = np.array(numeric_features_list)
    
    dropped_features = input_names[~mask]
    
    if len(dropped_features) > 0:
        logger.log_result(f"VarianceThreshold dropped {len(dropped_features)} features: {dropped_features.tolist()}")
    else:
        logger.log_result("VarianceThreshold did not drop any features.")

def transform(
    df: pd.DataFrame,
    subset: SubsetType,
    random_state: int,
    logger: MyLogger = DEF_NOTEBOOK_LOGGER,
    fitted_transformer: Path = FITTED_TRANSFORMER,
    print_to_console: bool = True,
    pandas_output: bool = False,
):
    logger.log_check("Performing df transformation...")
    transformed_array = None

    if subset == "train":
       
        logger.log_result(
            "Detected train subset. Creating new preprocessor...",
            print_to_console=print_to_console,
        )

        # 3. FIT and TRANSFORM
        transformer = build_transformer(random_state=random_state)

        transformer.fit(df)
        transformed_array = transformer.transform(df)


        log_dropped_features(transformer=transformer, numeric_features_list=NUMERIC_FEATURES + ENGINEERED_FEATURES, logger=logger)

        # 4. SAVE
        joblib.dump(transformer, fitted_transformer)

        # print("Fitted preprocessor saved to fitted_preprocessor.joblib")
    elif subset in ("test", "val"):
        logger.log_result(
            "Detected test subset. Loading fitted preprocessor...",
            print_to_console=print_to_console,
        )
        transformer: ColumnTransformer = joblib.load(fitted_transformer)

        # df = transformer.transform(df)
        transformed_array = transformer.transform(df)
    else:
        msg = "Unknown subset value!"
        logger.logger.error(msg)
        raise ValueError(msg)

    logger.log_result(
        "Transformations applied successfully.", print_to_console=print_to_console
    )

    if pandas_output:
        # transformer.set_output(transform="pandas") # <--- Force this specific instance
        # 2. Get the feature names
        # This works because your PrefixedTfidf implements get_feature_names_out
        feature_names = transformer.get_feature_names_out()

        # 3. Reconstruct the DataFrame
        df = pd.DataFrame(
            transformed_array, 
            columns=feature_names, 
            index=df.index  # Crucial to keep your original index!
        )
    else:
        df = transformed_array

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
        return float(pca.explained_variance_ratio_.sum())

    except KeyError as e:
        # raise KeyError(f"No PCA found under transformer '{name}'") from e
        # pass
        return -1

    # return float(pca.explained_variance_ratio_.sum())
