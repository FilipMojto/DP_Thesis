import argparse
from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Iterable, Literal
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


def get_n_jobs(reserve: int = 2) -> int:
    """
    Return number of cores to use, reserving some for system responsiveness.
    """
    total_cores = (os.cpu_count() / 2) or 1
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

# from pathlib import Path

# def get_experiment_dir(experiment_id: int, target_dir: Path) -> Path:
#     path = Path(f"{target_dir}/experiment_{experiment_id}")
#     path.mkdir(parents=True, exist_ok=True)
#     return path


# class MyParser(argparse.ArgumentParser):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def resolve_args(self, args: argparse.Namespace, resolvers: Dict[str, Callable[[Any], Any]]):
#         # args = self.parse_args()
#         return {name: resolver(args) for name, resolver in resolvers.items()}

#     def validate(
#         self,
#         args: argparse.Namespace,
#         validators: Iterable[Callable[[argparse.Namespace], None]],
#     ) -> None:
#         for validator in validators:
#             validator(args)

import os
import psutil
from dataclasses import dataclass, field
from typing import Literal

CoreModeType = Literal['manual', 'all']

@dataclass
class CoreConfig:
    reserve_cores: int = 2
    num_of_cores: int = 1
    mode: CoreModeType = 'manual'
    
    # Internal field to store the actual calculated value
    _final_n_jobs: int = field(init=False, repr=False)

    def __post_init__(self):
        self._calculate_cores()

    def _calculate_cores(self):
        # 1. Determine total physical availability
        # We use logical=False if you want physical cores, 
        # but for Grid Search, logical cores (Hyperthreading) are usually fine.
        total_available = os.cpu_count() or 1
        
        # Rule (a): Reserve cores must be complied with
        # We ensure we don't try to reserve more cores than exist
        actual_reserves = min(self.reserve_cores, total_available - 1)
        max_allowed = max(1, total_available - actual_reserves)

        if self.mode == 'all':
            # Rule (c) variant: If 'all', use everything except reserves
            self._final_n_jobs = max_allowed
        else:
            # Rule (b): Manual mode
            # Comply with manual setting only if it doesn't exceed the (Total - Reserved) limit
            if self.num_of_cores > max_allowed:
                self._final_n_jobs = max_allowed
            else:
                self._final_n_jobs = max(1, self.num_of_cores)

    @property
    def n_jobs(self) -> int:
        """This is what you pass to GridSearchCV(n_jobs=...)"""
        return self._final_n_jobs

    def __str__(self) -> str:
        total = os.cpu_count()
        return (
            f"--- Core Allocation Report ---\n"
            f"Mode:            {self.mode.upper()}\n"
            f"System Total:    {total} cores\n"
            f"Reserved:        {self.reserve_cores} cores\n"
            f"Target Manual:   {self.num_of_cores if self.mode == 'manual' else 'N/A'}\n"
            f"Final Allocated: {self._final_n_jobs} cores\n"
            f"------------------------------"
        )
