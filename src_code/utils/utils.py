import re

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

import time
from functools import wraps

def timeit(process_name: str = None):
    """
    Decorator to measure execution time of a function.
    
    Parameters:
        process_name (str): Optional name of the process to display.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = process_name or func.__name__
            print(f"[{name}] Starting...")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"[{name}] Finished in {elapsed:.4f} seconds.")
            return result
        return wrapper
    return decorator