import re

import pandas as pd

from notebooks.constants import ENGINEERED_FEATURES
from notebooks.logging_config import MyLogger

def is_embedding_column(col_name: str) -> bool:
    """
    Returns True if column name ends with emb_<positive_integer>
    Examples:
        emb_0        -> True
        code_emb_12 -> True
        msg_emb_768 -> True
        emb_        -> False
        emb_x       -> False
    """
    return bool(re.search(r"emb_\d+$", col_name))


def is_tfidf_vectorized(col_name: str) -> bool:
    """
    Returns True if column name represents a TF-IDF feature.

    Examples:
        tfidf_fix            -> True
        tfidf_fix_bug        -> True
        tfidf_memory_leak    -> True
        code_tfidf_fix       -> False
        tfidf                -> False
    """
    return bool(re.match(r"^tfidf_.+", col_name))


def is_engineered(col_name: str) -> bool:
    return col_name in ENGINEERED_FEATURES


def is_binary(df: pd.DataFrame, col_name: str) -> bool:
    values = df[col_name].dropna().unique()
    return len(values) == 2


import time
from functools import wraps

# def timeit(process_name: str = None):
#     """
#     Decorator to measure execution time of a function.
    
#     Parameters:
#         process_name (str): Optional name of the process to display.
#     """
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             name = process_name or func.__name__
#             print(f"[{name}] Starting...")
#             start_time = time.time()
#             result = func(*args, **kwargs)
#             end_time = time.time()
#             elapsed = end_time - start_time
#             print(f"[{name}] Finished in {elapsed:.4f} seconds.")
#             return result
#         return wrapper
#     return decorator

def timeit(process_name: str = None, logger_name: str = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Try to get name from the decorator param
            # 2. Try to get it from a specific keyword argument passed to the function
            # 3. Fall back to the function name
            name = process_name or kwargs.get("process_name") or func.__name__
            logger: MyLogger = None
            # print(f"[{name}] Starting...")
            if logger_name:
                logger = kwargs.get(logger_name)
                if logger:
                    logger.log_check(f"[{name}] Starting...")
            else:
                print(f"[{name}] Starting...")

            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            if logger:
                logger.log_result(f"[{name}] Finished in {elapsed:.4f} seconds.")
            else:
                print(f"[{name}] Finished in {elapsed:.4f} seconds.")

            return result
        return wrapper
    return decorator