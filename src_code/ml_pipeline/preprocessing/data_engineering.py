import pandas as pd

from typing import Callable, Dict, Iterable, Tuple

from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER


def create_derived_features(
    df: pd.DataFrame,
    mappings: Dict[str, Callable[[pd.DataFrame], pd.Series]],
    logger: NotebookLogger = DEF_NOTEBOOK_LOGGER,
) -> pd.DataFrame:
    logger.log_check("Deriving features...")

    for feature_name, func in mappings.items():
        try:
            result = func(df)
            if isinstance(result, pd.Series):
                df[feature_name] = result
                logger.log_result(f"Derived: {feature_name} - {func}")
            else:
                err_msg = f"Mapping for '{feature_name}' must return a pandas Series"
                logger.logger.error(err_msg)
                raise TypeError(err_msg)
        except KeyError:
            # silently skip if required columns are missing
            pass

    logger.log_result("Feature Derivation completed.")
    return df


Range = Tuple[float, float]
BucketMappings = Dict[str, Dict[Range, str]]


def create_buckets(
    df: pd.DataFrame,
    mappings: BucketMappings,
    logger: NotebookLogger = DEF_NOTEBOOK_LOGGER,
) -> pd.DataFrame:
    logger.log_check("Creating buckets...")
    for feature, submappings in mappings.items():
        if feature not in df.columns:
            continue

        # Sort ranges by start value
        sorted_ranges = sorted(submappings.items(), key=lambda x: x[0][0])

        # Build bins and labels
        bins = [r[0][0] for r in sorted_ranges]
        bins.append(sorted_ranges[-1][0][1])  # last upper bound

        labels = [label for _, label in sorted_ranges]

        df[f"{feature}_bucket"] = pd.cut(
            df[feature],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )
        logger.log_result(f"Created bucket for feature {feature}")

    logger.log_result("Buckets created successfully.")
    return df


def aggr_line_token_features(
    df: pd.DataFrame,
    features: Iterable[str],
    logger: NotebookLogger = DEF_NOTEBOOK_LOGGER,
) -> pd.DataFrame:
    logger.log_result("Aggregating line token features...")
    # features = list(features)  # ensure reusable iterable

    missing = [f for f in features if f not in df.columns]
    if missing:
        err_msg = f"Missing token columns: {missing}"
        logger.logger.error(err_msg)
        raise KeyError(err_msg)

    df["line_token_total"] = df[features].sum(axis=1)
    logger.log_result("Feature 'line_token_total' created successfully.")

    # Optionally create ratios per total lines (if loc_added exists)
    if "loc_added" in df.columns:
        logger.log_check("loc_added feature detected. Creating ratios per total lines.")
        denom = df["loc_added"] + 1  # avoid division by zero
        for token in features:
            df[f"{token}_ratio"] = df[token] / denom
            logger.log_result(f"{token}_ratio created successfully.")


    logger.log_result(("Aggregation successful."))
    return df


def create_feature_interactions(df: pd.DataFrame, features: Iterable[str]):
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    
    return df