from logging import Logger
from pathlib import Path
import pandas as pd
from typing import Callable, Dict, Iterable

import joblib
import numpy as np
from pandas import DataFrame, Series
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FunctionTransformer, Pipeline

from notebooks.logging_config import MyLogger
from notebooks.transformers import EmbeddingExpander, NamingPCA, WinsorizerIQR
from src_code.config import FITTED_TRANSFORMER, SubsetType
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.preprocessing.transform import build_transformer
# from src_code.ml_pipeline.utils import contains_negative


# def drop_negative_cols(
#     df: DataFrame,
#     numeric_features: Iterable[str],
#     # row_filters: Dict[str, bool],
#     row_filters: Dict[str, Callable[[Series], Series]],

#     # neg_features_to_drop: Iterable[str],

#     logger: NotebookLogger = DEF_NOTEBOOK_LOGGER,
#     print_to_console: bool = True,

# ):
#     logger.log_check(
#         "Dropping negative features and ensuring the rest is positive only...",
#         print_to_console=print_to_console,
#     )

#     if any(row not in numeric_features for row in row_filters):
#         err_msg = "Provided row not among the numeric features."
#         logger.logger.error(err_msg)
#         raise ValueError(err_msg)


#     neg_features_to_drop = row_filters.keys()
#     # List of features to check: NUMERIC_FEATURES excluding NEG_FEATURES_TO_DROP
#     features_to_check = [
#         col for col in numeric_features if col not in neg_features_to_drop
#     ]

#     # Check if any of the features in features_to_check contain negative values
#     if any(contains_negative(df, col) for col in features_to_check):
#         # If True, raise an exception
#         err_msg = "Unexpected negative values found in one or more numeric features that are NOT set to be dropped."
#         logger.logger.error(err_msg)
#         raise ValueError(err_msg)

#     neg_mask = df["time_since_last_change"] < 0
#     n_neg = neg_mask.sum()

#     logger.log_result(f"Dropping {n_neg} rows with negative time_since_last_change")

#     df = df[~neg_mask].reset_index(drop=True)

#     try:
#         assert any(contains_negative(df, col) for col in neg_features_to_drop) == False
#     except AssertionError as e:
#         logger.logger.error("One of filtered columns still contains negative rows.")


def drop_invalid_rows(
    df: DataFrame,
    # numeric_features: Iterable[str],
    row_filters: Dict[str, Callable[[Series], Series]],
    logger: MyLogger = DEF_NOTEBOOK_LOGGER,
    print_to_console: bool = True,
    sanity_check: bool = False,
) -> DataFrame:
    logger.log_check(
        "Applying row-level filters on numeric features...",
        print_to_console=print_to_console,
    )

    # Validate filter columns
    for col in row_filters:
        if col not in df.columns.to_list():
            raise ValueError(f"Filter column '{col}' is not a numeric feature.")

    # Build combined mask (AND across all filters)
    valid_mask = Series(True, index=df.index)

    for col, predicate in row_filters.items():
        col_mask = predicate(df[col])
        valid_mask &= col_mask

        # how many rows failed this filter, inversion is needed
        n_dropped = (~col_mask).sum()
        logger.log_result(f"Dropping {n_dropped} rows due to filter on '{col}'")

    # keeps only rows where the mask is true,the rest is dropped
    # the reset index removes gaps caused by dropped rows
    # By default, pandas tries to save your old index as a new column.
    # drop=True tells pandas to drop the old index instead of keeping it.
    df = df[valid_mask].reset_index(drop=True)

    if sanity_check:
        # Final sanity check (defensive programming)
        for col, predicate in row_filters.items():
            if not predicate(df[col]).all():
                raise AssertionError(
                    f"Filtering failed: column '{col}' still contains invalid rows."
                )

    return df


def drop_cols(
    df: pd.DataFrame, cols: Iterable[str], logger: MyLogger = DEF_NOTEBOOK_LOGGER
):
    logger.log_check("Dropping the specified columns...")

    start_cols = set(df.columns)

    df = df.drop(columns=cols, errors="ignore")

    end_cols = set(df.columns)

    logger.log_result("Dropping completed.")
    logger.log_result(f"Columns dropped: {len(start_cols - end_cols)}")
    logger.log_result(f"Columns remaining: {len(end_cols)}")
    return df

# def transform(
#     df: pd.DataFrame,
#     subset: SubsetType,
#     random_state: int,
#     logger: NotebookLogger = DEF_NOTEBOOK_LOGGER,
#     fitted_transformer: Path = FITTED_TRANSFORMER,
#     print_to_console: bool = True,
# ):
#     logger.log_check("Performing df transformation...")
#     # set_config(transform_output="pandas")
#     # log_transformer = FunctionTransformer(np.log1p, validate=False)

#     if subset == "train":
#         # log_check("Detected train subset. Creating new preprocessor...", print_to_console=True)
#         # preprocessor = ColumnTransformer(transformers=[], remainder='passthrough', verbose_feature_names_out=False)

#         # preprocessor.transformers.append(('winsorize', WinsorizerIQR(factor=1.5), NUMERIC_FEATURES))
#         # preprocessor.transformers.append(('log_tokens', log_transformer, LINE_TOKEN_FEATURES))
#         # preprocessor.transformers.append(('log_numeric', log_transformer, NUMERIC_FEATURES))

#         # # 3. FIT the preprocessor ONLY on the training data
#         # preprocessor.fit(df)
#         # df = preprocessor.transform(df)

#         # # 4. SAVE the fitted preprocessor
#         # # The saved object contains all the calculated Q1, Q3 bounds.
#         # joblib.dump(preprocessor, FITTED_PREPROCESSOR)
#         logger.log_result(
#             "Detected train subset. Creating new preprocessor...",
#             print_to_console=print_to_console,
#         )

#         # code_emb_df = expand_embedding(df, "code_embed", "code_emb")
#         # msg_emb_df  = expand_embedding(df, "msg_embed", "msg_emb")
#         # df = pd.concat([df.drop(columns=["code_embed", "msg_embed"]), code_emb_df, msg_emb_df], axis=1)

#         # Update the EMBEDDINGS constant to reflect the NEW flattened column names
#         # FLATTENED_EMBEDDINGS = code_emb_df.columns.tolist() + msg_emb_df.columns.tolist()

#         # Define a pipeline for EACH embedding type
#         # code_emb_pipe = Pipeline([
#         #     ('expand', EmbeddingExpander(prefix="code_emb")),
#         #     ('pca', PCA(n_components=100, random_state=RANDOM_STATE))
#         # ])
#         # Use it in your pipeline like this:
#         # code_emb_pipe = Pipeline(
#         #     [
#         #         ("expand", EmbeddingExpander(prefix="code")),
#         #         (
#         #             "pca",
#         #             NamingPCA(
#         #                 n_components=10, prefix="code_emb_", random_state=RANDOM_STATE
#         #             ),
#         #         ),
#         #     ]
#         # )

#         # msg_emb_pipe = Pipeline(
#         #     [
#         #         ("expand", EmbeddingExpander(prefix="msg")),
#         #         # ('pca', PCA(n_components=100, random_state=RANDOM_STATE))
#         #         (
#         #             "pca",
#         #             NamingPCA(
#         #                 n_components=45, prefix="msg_emb_", random_state=RANDOM_STATE
#         #             ),
#         #         ),
#         #     ]
#         # )

#         # # 1. Define a pipeline for numeric features: Winsorize THEN Log
#         # numeric_pipeline = Pipeline(
#         #     [
#         #         ("winsorize", WinsorizerIQR(factor=1.5)),
#         #         ("log", log_transformer),
#         #         ("var_thresh", VarianceThreshold(threshold=0.0)),
#         #     ]
#         # )

#         # # embedding_transformer = Pipeline(steps=[
#         # #     ("pca", PCA(n_components=100, random_state=RANDOM_STATE))
#         # # ])

#         # # 2. Setup the ColumnTransformer
#         # preprocessor = ColumnTransformer(
#         #     transformers=[
#         #         # ('num_transformed', numeric_pipeline, NUMERIC_FEATURES),
#         #         # ('token_transformed', log_transformer, LINE_TOKEN_FEATURES),
#         #         # ("embed", embedding_transformer, FLATTENED_EMBEDDINGS),
#         #         ("num", numeric_pipeline, NUMERIC_FEATURES),
#         #         ("tokens", log_transformer, LINE_TOKEN_FEATURES),
#         #         ("code_embed", code_emb_pipe, ["code_embed"]),  # Pass as list
#         #         ("msg_embed", msg_emb_pipe, ["msg_embed"]),  # Pass as list
#         #     ],
#         #     remainder="passthrough",
#         #     verbose_feature_names_out=False,  # This now works because names are unique
#         # )

#         # transformer_wrapper = TransformerInspector(random_state=random_state)
#         # transformer = transformer_wrapper.transformer

#         # 3. FIT and TRANSFORM
#         transformer = build(random_state=random_state)

#         transformer.fit(df)
#         df = transformer.transform(df)

#         # 4. SAVE
#         joblib.dump(transformer, fitted_transformer)

#         # print("Fitted preprocessor saved to fitted_preprocessor.joblib")
#     elif subset in ("test", "validate"):
#         logger.log_result(
#             "Detected test subset. Loading fitted preprocessor...",
#             print_to_console=print_to_console,
#         )
#         transformer: ColumnTransformer = joblib.load(fitted_transformer)
#         # transformer_wrapper = TransformerInspector(
#         #     random_state=random_state, fitted_transformer=transformer_wrapper
#         # )
#         df = transformer.transform(df)
#     else:
#         msg = "Unknown subset value!"
#         logger.logger.error(msg)
#         raise ValueError(msg)

#     logger.log_result(
#         "Transformations applied successfully.", print_to_console=print_to_console
#     )

#     return df, transformer
