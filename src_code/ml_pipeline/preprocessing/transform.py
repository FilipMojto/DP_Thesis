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
from notebooks.logging_config import NotebookLogger
from notebooks.transformers import EmbeddingExpander, NamingPCA, WinsorizerIQR
from src_code.config import FITTED_TRANSFORMER, SubsetType
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER


set_config(transform_output="pandas")
log_transformer = FunctionTransformer(np.log1p, validate=False)


def build(random_state: int):
    pipelines: List[Pipeline] = []

    code_emb_pipe = Pipeline(
        [
            ("expand", EmbeddingExpander(prefix="code")),
            (
                "pca",
                NamingPCA(
                    n_components=10, prefix="code_emb_", random_state=random_state
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
                    n_components=45, prefix="msg_emb_", random_state=random_state
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

    pipelines.extend([msg_emb_pipe, code_emb_pipe, numeric_pipe])

    # embedding_transformer = Pipeline(steps=[
    #     ("pca", PCA(n_components=100, random_state=RANDOM_STATE))
    # ])

    # 2. Setup the ColumnTransformer
    # self.verb
    transformer = ColumnTransformer(
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

    return transformer


# class TransformerInspector:

#     def __init__(self, fitted_transformer: ColumnTransformer = None):
#         # self.pipelines: List[Pipeline] = []
#         # self.random_state = random_state
#         # np.random.seed(random_state)

#         # if fitted_transformer is None:
#         #     code_emb_pipe = Pipeline(
#         #         [
#         #             ("expand", EmbeddingExpander(prefix="code")),
#         #             (
#         #                 "pca",
#         #                 NamingPCA(
#         #                     n_components=10,
#         #                     prefix="code_emb_",
#         #                     random_state=self.random_state,
#         #                 ),
#         #             ),
#         #         ]
#         #     )

#         #     msg_emb_pipe = Pipeline(
#         #         [
#         #             ("expand", EmbeddingExpander(prefix="msg")),
#         #             # ('pca', PCA(n_components=100, random_state=RANDOM_STATE))
#         #             (
#         #                 "pca",
#         #                 NamingPCA(
#         #                     n_components=45,
#         #                     prefix="msg_emb_",
#         #                     random_state=self.random_state,
#         #                 ),
#         #             ),
#         #         ]
#         #     )

#         #     # 1. Define a pipeline for numeric features: Winsorize THEN Log
#         #     numeric_pipe = Pipeline(
#         #         [
#         #             ("winsorize", WinsorizerIQR(factor=1.5)),
#         #             ("log", log_transformer),
#         #             ("var_thresh", VarianceThreshold(threshold=0.0)),
#         #         ]
#         #     )

#         #     self.pipelines.extend([msg_emb_pipe, code_emb_pipe, numeric_pipe])

#         #     # embedding_transformer = Pipeline(steps=[
#         #     #     ("pca", PCA(n_components=100, random_state=RANDOM_STATE))
#         #     # ])

#         #     # 2. Setup the ColumnTransformer
#         #     # self.verb
#         #     self.transformer = ColumnTransformer(
#         #         transformers=[
#         #             # ('num_transformed', numeric_pipeline, NUMERIC_FEATURES),
#         #             # ('token_transformed', log_transformer, LINE_TOKEN_FEATURES),
#         #             # ("embed", embedding_transformer, FLATTENED_EMBEDDINGS),
#         #             ("num", numeric_pipe, NUMERIC_FEATURES),
#         #             ("tokens", log_transformer, LINE_TOKEN_FEATURES),
#         #             ("code_embed", code_emb_pipe, ["code_embed"]),  # Pass as list
#         #             ("msg_embed", msg_emb_pipe, ["msg_embed"]),  # Pass as list
#         #         ],
#         #         remainder="passthrough",
#         #         verbose_feature_names_out=False,  # This now works because names are unique
#         #     )
#         # else:
#         self.transformer = fitted_transformer

def transform(
    df: pd.DataFrame,
    subset: SubsetType,
    random_state: int,
    logger: NotebookLogger = DEF_NOTEBOOK_LOGGER,
    fitted_transformer: Path = FITTED_TRANSFORMER,
    print_to_console: bool = True,
):
    logger.log_check("Performing df transformation...")
    # set_config(transform_output="pandas")
    # log_transformer = FunctionTransformer(np.log1p, validate=False)

    if subset == "train":
        # log_check("Detected train subset. Creating new preprocessor...", print_to_console=True)
        # preprocessor = ColumnTransformer(transformers=[], remainder='passthrough', verbose_feature_names_out=False)

        # preprocessor.transformers.append(('winsorize', WinsorizerIQR(factor=1.5), NUMERIC_FEATURES))
        # preprocessor.transformers.append(('log_tokens', log_transformer, LINE_TOKEN_FEATURES))
        # preprocessor.transformers.append(('log_numeric', log_transformer, NUMERIC_FEATURES))

        # # 3. FIT the preprocessor ONLY on the training data
        # preprocessor.fit(df)
        # df = preprocessor.transform(df)

        # # 4. SAVE the fitted preprocessor
        # # The saved object contains all the calculated Q1, Q3 bounds.
        # joblib.dump(preprocessor, FITTED_PREPROCESSOR)
        logger.log_result(
            "Detected train subset. Creating new preprocessor...",
            print_to_console=print_to_console,
        )

        # code_emb_df = expand_embedding(df, "code_embed", "code_emb")
        # msg_emb_df  = expand_embedding(df, "msg_embed", "msg_emb")
        # df = pd.concat([df.drop(columns=["code_embed", "msg_embed"]), code_emb_df, msg_emb_df], axis=1)

        # Update the EMBEDDINGS constant to reflect the NEW flattened column names
        # FLATTENED_EMBEDDINGS = code_emb_df.columns.tolist() + msg_emb_df.columns.tolist()

        # Define a pipeline for EACH embedding type
        # code_emb_pipe = Pipeline([
        #     ('expand', EmbeddingExpander(prefix="code_emb")),
        #     ('pca', PCA(n_components=100, random_state=RANDOM_STATE))
        # ])
        # Use it in your pipeline like this:
        # code_emb_pipe = Pipeline(
        #     [
        #         ("expand", EmbeddingExpander(prefix="code")),
        #         (
        #             "pca",
        #             NamingPCA(
        #                 n_components=10, prefix="code_emb_", random_state=RANDOM_STATE
        #             ),
        #         ),
        #     ]
        # )

        # msg_emb_pipe = Pipeline(
        #     [
        #         ("expand", EmbeddingExpander(prefix="msg")),
        #         # ('pca', PCA(n_components=100, random_state=RANDOM_STATE))
        #         (
        #             "pca",
        #             NamingPCA(
        #                 n_components=45, prefix="msg_emb_", random_state=RANDOM_STATE
        #             ),
        #         ),
        #     ]
        # )

        # # 1. Define a pipeline for numeric features: Winsorize THEN Log
        # numeric_pipeline = Pipeline(
        #     [
        #         ("winsorize", WinsorizerIQR(factor=1.5)),
        #         ("log", log_transformer),
        #         ("var_thresh", VarianceThreshold(threshold=0.0)),
        #     ]
        # )

        # # embedding_transformer = Pipeline(steps=[
        # #     ("pca", PCA(n_components=100, random_state=RANDOM_STATE))
        # # ])

        # # 2. Setup the ColumnTransformer
        # preprocessor = ColumnTransformer(
        #     transformers=[
        #         # ('num_transformed', numeric_pipeline, NUMERIC_FEATURES),
        #         # ('token_transformed', log_transformer, LINE_TOKEN_FEATURES),
        #         # ("embed", embedding_transformer, FLATTENED_EMBEDDINGS),
        #         ("num", numeric_pipeline, NUMERIC_FEATURES),
        #         ("tokens", log_transformer, LINE_TOKEN_FEATURES),
        #         ("code_embed", code_emb_pipe, ["code_embed"]),  # Pass as list
        #         ("msg_embed", msg_emb_pipe, ["msg_embed"]),  # Pass as list
        #     ],
        #     remainder="passthrough",
        #     verbose_feature_names_out=False,  # This now works because names are unique
        # )

        # transformer_wrapper = TransformerInspector(random_state=random_state)
        # transformer = transformer_wrapper.transformer

        # 3. FIT and TRANSFORM
        transformer = build(random_state=random_state)

        transformer.fit(df)
        df = transformer.transform(df)

        # 4. SAVE
        joblib.dump(transformer, fitted_transformer)

        # print("Fitted preprocessor saved to fitted_preprocessor.joblib")
    elif subset in ("test", "validate"):
        logger.log_result(
            "Detected test subset. Loading fitted preprocessor...",
            print_to_console=print_to_console,
        )
        transformer: ColumnTransformer = joblib.load(fitted_transformer)
        # transformer_wrapper = TransformerInspector(
        #     random_state=random_state, fitted_transformer=transformer_wrapper
        # )
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
