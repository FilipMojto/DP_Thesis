
from typing import Dict, get_args

import pandas as pd

from notebooks.logging_config import MyLogger
from src_code.config import JIT_DIR, LOG_DIR, SubsetType
from src_code.ml_pipeline.data_utils import load_df


def check_commit_year_distribution(df: pd.DataFrame, logger: MyLogger):
    if "datetime" not in df.columns:
        logger.log_result("Column 'datetime' not found in DataFrame. Skipping commit year distribution check.")
        return
    
    # Extract year from datetime column
    df["commit_year"] = pd.to_datetime(df["datetime"], errors="coerce").dt.year
    year_counts = df["commit_year"].value_counts().sort_index()
    logger.log_result(f"Commit year distribution:\n{year_counts}", print_to_console=True)


# def compare_train_test_means(X_train: pd.DataFrame, X_test: pd.DataFrame):
#     for col in X_train.columns:
#         if X_train[col].dtype != "object":
#             print(
#                 col,
#                 X_train[col].mean(),
#                 X_test[col].mean()
#         )

if __name__ == "__main__":
    logger = MyLogger(label="extract", section_name="extract", file_log_path=LOG_DIR / "extract.log")
    subset_types = get_args(SubsetType)
    input_dfs: Dict[str, pd.DataFrame] = {}
    
    # input_dfs = load_df(df_file_path=JIT_DIR / "train.feather", logger=logger)

    for subset in subset_types:
        input_dfs[subset] = load_df(df_file_path=JIT_DIR / f"{subset}.feather", logger=logger)

    input_df = input_dfs["train"]  # Check distribution on training set as it's the largest and most representative

    check_commit_year_distribution(df=input_df, logger=logger)
    compare_train_test_means(X_train=input_dfs["train"].drop(columns=["target"]), X_test=input_dfs["test"].drop(columns=["target"]))