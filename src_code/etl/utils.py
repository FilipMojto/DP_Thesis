from typing import Iterable
from git import Repo
import pandas as pd

from src_code.config import PYTHON_LIBS_DIR


def get_repo_instance(repo_name: str) -> Repo:
    """Helper function to load the local Git Repo object."""
    # Assuming PYTHON_LIBS_DIR points to where repos are cloned
    return Repo(PYTHON_LIBS_DIR / repo_name)


def find_duplicates(df: pd.DataFrame, keys: Iterable[str]):
    dupes = df.duplicated(subset=keys, keep=False)

    duplicate_rows = df.index[dupes]
    return duplicate_rows.tolist()