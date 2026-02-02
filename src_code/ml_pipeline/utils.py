import os
import pandas as pd

from notebooks.logging_config import MyLogger
from src_code.config import EVALUATION_DIR, REPORTS_DIR

# def contains_negative(df: pd.DataFrame, col: str) -> bool:
#     """
#     Checks if a specified numeric column in a DataFrame contains at least one negative value.

#     Args:
#         df: The pandas DataFrame to check.
#         col: The name of the column to inspect.

#     Returns:
#         True if the column contains one or more negative values; False otherwise.
#     """
    
#     # 1. Check if the column exists
#     if col not in df.columns:
#         raise ValueError(f"Column '{col}' not found in the DataFrame.")
    
#     # 2. Check if the column is numeric (optional, but good practice)
#     # The comparison (df[col] < 0) will work on non-numeric types but raise a warning
#     # or unexpected results. This check makes it robust.
#     if not pd.api.types.is_numeric_dtype(df[col]):
#         print(f"Warning: Column '{col}' is not a numeric type. Proceeding with comparison.")
    
#     # 3. Core Logic: Check for any value less than 0
#     # Create a boolean series where True indicates a negative value
#     is_negative = (df[col] < 0)
    
#     # .any() returns True if there is at least one True in the boolean series
#     return is_negative.any()


def get_n_jobs(reserve: int = 6) -> int:
    """
    Return number of cores to use, reserving some for system responsiveness.
    """
    total_cores = os.cpu_count() or 1
    return max(1, total_cores - reserve)


def limit_dataframe_rows(df: pd.DataFrame, script_logger: MyLogger, max_rows: int = None) -> pd.DataFrame:
    if max_rows is not None:
        script_logger.log_check(f"Limiting to first {max_rows} rows for testing...")
        target_df = df.head(max_rows)
        script_logger.log_result(
            f"Dataframe shape after row limit: {target_df.shape}",
            print_to_console=True,
        )
        return target_df
    return df


def describe_dataframe(df: pd.DataFrame, logger: MyLogger, name: str = "DataFrame"):
    logger.log_result(f"Describing {name}...")
    # desc = df.describe(include="all").T
    # logger.log_result(f"\n{desc}")
    logger.log_result(
        f"Initial dataframe shape: {df.shape}", print_to_console=True
    )
    # return desc

from pathlib import Path

def get_experiment_dir(experiment_id: int) -> Path:
    path = Path(f"{EVALUATION_DIR}/experiment_{experiment_id}")
    path.mkdir(parents=True, exist_ok=True)
    return path