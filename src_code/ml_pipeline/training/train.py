import pandas as pd
from sklearn.model_selection import train_test_split

from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER


def split_train_test(
    df: pd.DataFrame,
    target: str,
    random_state: int,
    test_size: float,
    logger: NotebookLogger = DEF_NOTEBOOK_LOGGER,
):
    logger.log_check("Splitting df into train & test subsets...")
    n_rows = len(df)
    n_features = df.shape[1] - 1  # excluding target

    logger.log_result(f"Total rows before split: {n_rows}")
    logger.log_result(f"Feature count (X): {n_features}")
    logger.log_result(f"Target column: '{target}'")
    logger.log_result(f"Test size: {test_size:.2%}")


    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        # stratify=y, not required since the training subset of the original df is balanced
        random_state=random_state,
    )

    logger.log_result(
        f"Train rows: {len(X_train)} | Test rows: {len(X_test)}"
    )

    logger.log_result("Splitting completed.")
    return X_train, X_test, y_train, y_test
