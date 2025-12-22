from logging import Logger
from typing import Callable, Dict, Iterable

from pandas import DataFrame, Series

from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.utils import contains_negative


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
    numeric_features: Iterable[str],
    row_filters: Dict[str, Callable[[Series], Series]],
    logger: NotebookLogger = DEF_NOTEBOOK_LOGGER,
    print_to_console: bool = True,
) -> DataFrame:
    logger.log_check(
        "Applying row-level filters on numeric features...",
        print_to_console=print_to_console,
    )

    # Validate filter columns
    for col in row_filters:
        if col not in numeric_features:
            raise ValueError(f"Filter column '{col}' is not a numeric feature.")

    # Build combined mask (AND across all filters)
    valid_mask = Series(True, index=df.index)

    for col, predicate in row_filters.items():
        col_mask = predicate(df[col])
        valid_mask &= col_mask

        n_dropped = (~col_mask).sum()
        logger.log_result(
            f"Dropping {n_dropped} rows due to filter on '{col}'"
        )

    df = df[valid_mask].reset_index(drop=True)

    # Final sanity check
    for col, predicate in row_filters.items():
        if not predicate(df[col]).all():
            raise AssertionError(
                f"Filtering failed: column '{col}' still contains invalid rows."
            )

    return df