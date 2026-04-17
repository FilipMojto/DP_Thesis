from __future__ import annotations
import argparse
from typing import Dict, get_args

import numpy as np
import pandas as pd

from notebooks.logging_config import MyLogger
from src_code.config import EXTRACTED_DFS, JIT_DIR, LOG_DIR, SubsetType
from src_code.ml_pipeline.data_utils import load_df, save_df



import re
from pathlib import PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from src_code.ml_pipeline.utils import limit_dataframe_rows
from src_code.versioning import VersionedFileManager


# -----------------------------
# Small safe helpers
# -----------------------------

def _safe_list_len(value) -> int:
    """Return len(value) if it looks like a list, otherwise 0."""
    if isinstance(value, list):
        return len(value)
    if pd.isna(value):
        return 0
    return 0


# def _safe_list(value) -> list:
#     """Return a list or an empty list."""
#     if isinstance(value, list):
#         return value
#     return []
def _safe_list(value) -> list:
    """Return a list from list/array-like input, otherwise empty list."""
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return []
    return []


def _count_lines(text: str) -> int:
    if not isinstance(text, str) or text == "":
        return 0
    return len(text.splitlines())


def _is_comment_line(code_line: str) -> bool:
    stripped = code_line.strip()
    return stripped.startswith("#")


def _is_blank_line(code_line: str) -> bool:
    return code_line.strip() == ""


def _looks_like_diff_header(line: str) -> bool:
    return (
        line.startswith("diff --git ")
        or line.startswith("--- ")
        or line.startswith("+++ ")
        or line.startswith("index ")
        or line.startswith("new file mode ")
        or line.startswith("deleted file mode ")
        or line.startswith("rename from ")
        or line.startswith("rename to ")
    )


def _parse_unified_diff(diff_text: str) -> Dict[str, int]:
    """
    Parse a unified diff string and return basic churn features.
    Works on typical git diff text.
    """
    stats = {
        "loc_added": 0,
        "loc_deleted": 0,
        "context_lines": 0,
        "hunk_count": 0,
        "diff_line_count": 0,
        "added_comment_lines": 0,
        "deleted_comment_lines": 0,
        "added_blank_lines": 0,
        "deleted_blank_lines": 0,
        "added_code_lines": 0,
        "deleted_code_lines": 0,
    }

    if not isinstance(diff_text, str) or diff_text.strip() == "":
        return stats

    for raw_line in diff_text.splitlines():
        stats["diff_line_count"] += 1

        if _looks_like_diff_header(raw_line):
            continue

        if raw_line.startswith("@@"):
            stats["hunk_count"] += 1
            continue

        # Ignore file path markers that are part of diff metadata
        if raw_line.startswith("\\ No newline at end of file"):
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            line = raw_line[1:]
            stats["loc_added"] += 1
            if _is_blank_line(line):
                stats["added_blank_lines"] += 1
            elif _is_comment_line(line):
                stats["added_comment_lines"] += 1
            else:
                stats["added_code_lines"] += 1

        elif raw_line.startswith("-") and not raw_line.startswith("---"):
            line = raw_line[1:]
            stats["loc_deleted"] += 1
            if _is_blank_line(line):
                stats["deleted_blank_lines"] += 1
            elif _is_comment_line(line):
                stats["deleted_comment_lines"] += 1
            else:
                stats["deleted_code_lines"] += 1

        else:
            stats["context_lines"] += 1

    return stats


# def _extract_path_features(filepath: str) -> Dict[str, object]:
#     """
#     Extract file-path based predictors.
#     """
#     if not isinstance(filepath, str) or filepath.strip() == "":
#         return {
#             "path_depth": 0,
#             "filename_len": 0,
#             "stem_len": 0,
#             "ext": "",
#             "is_test_file": 0,
#             "is_init_file": 0,
#             "has_src_dir": 0,
#             "has_docs_dir": 0,
#             "has_tests_dir": 0,
#         }

#     path = PurePosixPath(filepath.replace("\\", "/"))
#     filename = path.name
#     stem = path.stem.lower()
#     parts = [p.lower() for p in path.parts]

#     ext = path.suffix.lower().lstrip(".")
#     return {
#         "path_depth": max(len(path.parts) - 1, 0),
#         "filename_len": len(filename),
#         "stem_len": len(path.stem),
#         "ext": ext,
#         "is_test_file": int(
#             "test" in stem
#             or "tests" in parts
#             or any(part.startswith("test") for part in parts)
#         ),
#         "is_init_file": int(filename == "__init__.py"),
#         "has_src_dir": int("src" in parts),
#         "has_docs_dir": int("docs" in parts),
#         "has_tests_dir": int("tests" in parts),
#     }
def _extract_path_features(filepath: str) -> Dict[str, object]:
    """
    Extract file-path based predictors (robust version).
    """

    if not isinstance(filepath, str) or filepath.strip() == "":
        return {
            "path_depth": 0,
            "filename_len": 0,
            "stem_len": 0,
            "ext": "",
            "is_test_file": 0,
            "is_init_file": 0,
            "has_src_dir": 0,
            "has_docs_dir": 0,
            # "has_tests_dir": 0,
        }

    path = PurePosixPath(filepath.replace("\\", "/"))
    filename = path.name
    stem = path.stem.lower()
    parts = [p.lower() for p in path.parts]
    ext = path.suffix.lower().lstrip(".")

    # --- TEST FILE DETECTION (improved) ---
    test_dir_markers = {
        "test", "tests", "__tests__", "testing", "spec", "specs"
    }

    test_file_patterns = (
        stem.startswith("test_")
        or stem.endswith("_test")
        or stem.endswith("_tests")
        or "_test_" in stem
        or "_spec" in stem
        or stem.endswith("spec")
    )

    is_test_file = int(
        test_file_patterns
        or any(part in test_dir_markers for part in parts)
        or any(part.startswith("test") for part in parts)
    )

    # --- directory features ---
    has_src_dir = int("src" in parts)
    has_docs_dir = int("docs" in parts)

    # has_tests_dir = int(
    #     any(part in test_dir_markers for part in parts)
    # )

    return {
        "path_depth": max(len(path.parts) - 1, 0),
        "filename_len": len(filename),
        "stem_len": len(path.stem),
        "ext": ext,
        "is_test_file": is_test_file,
        "is_init_file": int(filename == "__init__.py"),
        "has_src_dir": has_src_dir,
        "has_docs_dir": has_docs_dir,
        # "has_tests_dir": has_tests_dir,
    }


def _extract_datetime_features(dt: pd.Series) -> pd.DataFrame:
    """
    Convert commit datetime into calendar/time predictors.
    """
    dt = pd.to_datetime(dt, errors="coerce")

    out = pd.DataFrame(index=dt.index)
    out["commit_year"] = dt.dt.year
    out["commit_month"] = dt.dt.month
    out["commit_day"] = dt.dt.day
    out["commit_dayofweek"] = dt.dt.dayofweek
    out["commit_hour"] = dt.dt.hour
    out["commit_is_weekend"] = (dt.dt.dayofweek >= 5).astype("Int64")
    out["commit_is_month_start"] = dt.dt.is_month_start.astype("Int64")
    out["commit_is_month_end"] = dt.dt.is_month_end.astype("Int64")
    out["commit_is_quarter_end"] = dt.dt.is_quarter_end.astype("Int64")
    return out


def _extract_method_features(methods_col: pd.Series) -> pd.DataFrame:
    """
    Extract method-level predictors from the list of changed methods.
    """
    rows = []
    for value in methods_col:
        methods = _safe_list(value)
        lengths = [len(str(m)) for m in methods if m is not None]

        rows.append(
            {
                "changed_methods_count": len(methods),
                "changed_methods_avg_name_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
                "changed_methods_max_name_len": max(lengths) if lengths else 0,
                "changed_methods_has_dunder": int(any(str(m).startswith("__") for m in methods)),
            }
        )

    return pd.DataFrame(rows, index=methods_col.index)


def _extract_diff_features(content_col: pd.Series) -> pd.DataFrame:
    """
    Extract churn and diff-structure predictors from git diff content.
    """
    rows = []
    for text in content_col:
        diff_stats = _parse_unified_diff(text if isinstance(text, str) else "")
        total_changed = diff_stats["loc_added"] + diff_stats["loc_deleted"]

        diff_len = _count_lines(text if isinstance(text, str) else "")
        rows.append(
            {
                **diff_stats,
                "total_changed_loc": total_changed,
                "net_loc_change": diff_stats["loc_added"] - diff_stats["loc_deleted"],
                "change_intensity": (total_changed / diff_len) if diff_len else 0.0,
                "diff_char_len": len(text) if isinstance(text, str) else 0,
                "diff_line_density": (diff_len / max(len(text), 1)) if isinstance(text, str) else 0.0,
            }
        )

    return pd.DataFrame(rows, index=content_col.index)


# def _extract_label_features(lines_col: pd.Series) -> pd.DataFrame:
#     """
#     Create the target column from the line-level bug annotation.
#     Do not use this as a predictor.
#     """
#     rows = []
#     for value in lines_col:
#         lines = _safe_list(value)
#         rows.append(
#             {
#                 "target": int(len(lines) > 0),
#             }
#         )
#     return pd.DataFrame(rows, index=lines_col.index)
# def _extract_label_features(lines_col: pd.Series) -> pd.DataFrame:
#     rows = []

#     for value in lines_col:
#         if isinstance(value, (list, np.ndarray)):
#             target = int(len(value) > 0)
#         else:
#             target = 0

#         rows.append({"target": target})

#     return pd.DataFrame(rows, index=lines_col.index)
def _extract_label_features(lines_col: pd.Series) -> pd.DataFrame:
    target = lines_col.apply(
        lambda x: int(len(x) > 0) if isinstance(x, (list, np.ndarray)) else 0
    )
    return pd.DataFrame({"target": target}, index=lines_col.index)


def _extract_repo_features(repo_col: pd.Series) -> pd.DataFrame:
    """
    Optional repository identifier features.
    Keep repo as a raw column too; later you can one-hot encode or drop it.
    """
    out = pd.DataFrame(index=repo_col.index)
    out["repo_name_len"] = repo_col.fillna("").astype(str).str.len()
    # out["repo_has_slash"] = repo_col.fillna("").astype(str).str.contains("/", regex=False).astype("Int64")
    return out


# -----------------------------
# Main feature table builder
# -----------------------------

def build_jit_file_level_dataset(
    raw_df: pd.DataFrame,
    keep_raw_columns: bool = True,
    extra_extractors: Optional[List[Callable[[pd.DataFrame], pd.DataFrame]]] = None,
) -> pd.DataFrame:
    """
    Transform the raw Defectors JIT split into a modeling-ready table.

    Parameters
    ----------
    raw_df:
        Input dataframe with columns like:
        datetime, commit, repo, filepath, content, methods, lines

    keep_raw_columns:
        If True, preserve original columns for EDA and traceability.
        If False, return only engineered features + target.

    extra_extractors:
        Optional list of custom extractor functions.
        Each function must accept the full dataframe and return a dataframe
        with the same index.

    Returns
    -------
    pd.DataFrame
        Feature table ready for EDA and classical ML.
    """
    df = raw_df.copy()

    # Normalize datetime once
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    feature_frames: List[pd.DataFrame] = []

    # Label first
    if "lines" not in df.columns:
        raise ValueError("Expected a 'lines' column for target creation.")
    feature_frames.append(_extract_label_features(df["lines"]))

    # Core feature families
    if "datetime" in df.columns:
        feature_frames.append(_extract_datetime_features(df["datetime"]))

    if "filepath" in df.columns:
        path_features = df["filepath"].apply(_extract_path_features).apply(pd.Series)
        feature_frames.append(path_features)

    if "methods" in df.columns:
        feature_frames.append(_extract_method_features(df["methods"]))

    if "content" in df.columns:
        feature_frames.append(_extract_diff_features(df["content"]))

    if "repo" in df.columns:
        feature_frames.append(_extract_repo_features(df["repo"]))

    # Any future extractors can be plugged in here
    if extra_extractors:
        for extractor in extra_extractors:
            feature_frames.append(extractor(df))

    features = pd.concat(feature_frames, axis=1)

    if keep_raw_columns:
        # Keep the original data for EDA / tracing / debugging
        out = pd.concat([df, features], axis=1)
    else:
        out = features

    # Recommended cleanup: remove duplicated columns if any extractor overlaps
    out = out.loc[:, ~out.columns.duplicated()]

    return out


# -----------------------------
# Example of a future AST extractor
# -----------------------------
def extract_ast_proxy_features_from_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for a future AST-oriented feature family.

    True AST metrics generally require before/after source code, not just a diff.
    For the current JIT diff setting, you can use this as a stub or replace it
    later with a real parser-based extractor.

    Example real AST features later:
    - ast_node_count_before
    - ast_node_count_after
    - ast_node_delta
    - function_def_delta
    - class_def_delta
    - cyclomatic_complexity_delta
    """
    return pd.DataFrame(
        {
            "ast_node_count_before": pd.NA,
            "ast_node_count_after": pd.NA,
            "ast_node_delta": pd.NA,
            "function_def_delta": pd.NA,
            "class_def_delta": pd.NA,
            "cyclomatic_complexity_delta": pd.NA,
        },
        index=df.index,
    )

if __name__ == "__main__":

    # let's load raw data first

    parser = argparse.ArgumentParser(description="Extract features from raw JIT diff data.")
    parser.add_argument(
        "--max-rows",
        type=int,
        required=False,
        default=None,
        help="Limit dataset to first n rows only for testing purposes.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=get_args(SubsetType),
        default="train",
        required=False,
        help="Specify which subset (train, test or validate) to run through the pipelin e.",
    )

    args = parser.parse_args()
    subset_types = get_args(SubsetType)

    logger = MyLogger(label="extract", section_name="extract", file_log_path=LOG_DIR / "extract.log")
    
    input_dfs: Dict[str, pd.DataFrame] = {}

    # input_dfs = load_df(df_file_path=JIT_DIR / "train.feather", logger=logger)

    for subset in subset_types:
        input_dfs[subset] = load_df(df_file_path=JIT_DIR / f"{subset}.feather", logger=logger)

    analyzed_subset = args.subset
    input_df = input_dfs[analyzed_subset]

    # inspect the dataframe
    logger.log_result(f"Total rows in all datasets: {sum(df.shape[0] for df in input_dfs.values())}")
    logger.log_result(f"Input dataframe shape: {input_df.shape}")
    logger.log_result(f"Input dataframe columns: {input_df.columns.tolist()}")
    logger.log_result(f"Input dataframe dtypes:\n{input_df.dtypes}")

    # limit rows for testing
    # input_df = limit_dataframe_rows(df=input_df, script_logger=logger, max_rows=args.max_rows)
    # input_df = input_df.sample(n=args.max_rows, stratify=input_df["target"])
    if args.max_rows:
        input_df = input_df.sample(n=args.max_rows, random_state=42)
        logger.log_result(f"Dataset limited to {args.max_rows} rows for testing.")
        logger.log_result(f"Dataset limited to first {args.max_rows} rows for testing.")

    prepared_df = build_jit_file_level_dataset(
        raw_df=input_df,
        keep_raw_columns=True,
        extra_extractors=[extract_ast_proxy_features_from_diff],
    )

    logger.log_result(f"Prepared dataframe shape: {prepared_df.shape}")
    logger.log_result(f"Prepared dataframe columns: {prepared_df.columns.tolist()}")
    logger.log_result(f"Prepared dataframe dtypes:\n{prepared_df.dtypes}")

    prev_df = VersionedFileManager(file_path=EXTRACTED_DFS / f"{analyzed_subset}_extracted.feather", logger=logger)
    save_df(df=prepared_df, df_file_path=prev_df.next_base_output, logger=logger)