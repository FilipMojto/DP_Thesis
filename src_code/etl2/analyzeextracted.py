from typing import Dict, get_args

import pandas as pd

from notebooks.logging_config import MyLogger
from src_code.config import EXTRACTED_DFS, JIT_DIR, LOG_DIR, SubsetType
from src_code.etl2.analyzeraw import check_commit_year_distribution
from src_code.ml_pipeline.data_utils import load_df
from src_code.versioning import VersionedFileManager


def compare_train_test_means(X_train: pd.DataFrame, X_test: pd.DataFrame):
    for col in X_train.columns:
        if X_train[col].dtype not in ["object", "datetime64[ns]", "string"]:
            print(
                col,
                X_train[col].mean(),
                X_test[col].mean()
        )
    

def compare_target_distribution(y_train: pd.Series, y_test: pd.Series):
    print("Target distribution in training set:")
    print(y_train.value_counts(normalize=True))
    print("\nTarget distribution in test set:")
    print(y_test.value_counts(normalize=True))

    
            

if __name__ == "__main__":
    logger = MyLogger(label="extract", section_name="extract", file_log_path=LOG_DIR / "extract.log")
    subset_types = get_args(SubsetType)
    input_dfs: Dict[str, pd.DataFrame] = {}
    
    # input_dfs = load_df(df_file_path=JIT_DIR / "train.feather", logger=logger)

    for subset in subset_types:
        input_dfs[subset] = load_df(df_file_path=VersionedFileManager(EXTRACTED_DFS / f"{subset}_extracted.feather", logger=logger).current_newest, logger=logger)

    input_df = input_dfs["train"]  # Check distribution on training set as it's the largest and most representative

    # check_commit_year_distribution(df=input_df, logger=logger)
    compare_train_test_means(X_train=input_dfs["train"].drop(columns=["target"]), X_test=input_dfs["test"].drop(columns=["target"]))
    compare_target_distribution(y_train=input_dfs["train"]["target"], y_test=input_dfs["test"]["target"])