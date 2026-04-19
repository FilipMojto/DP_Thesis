from __future__ import annotations

from argparse import ArgumentParser
import math
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pandas as pd

from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERED_DATA_DIR, EXTENDED_DATA_DIR, LOG_DIR, SubsetType
from src_code.ml_pipeline.data_utils import load_df
from src_code.versioning import VersionedFileManager

parser = ArgumentParser(description="Validates data cleanness of extracted data.")
parser.add_argument(
    "--subset",
    # choices=["train", "test", "validate"],  # This is the key part
    choices=get_args(SubsetType),
    required=False,  # Recommend making it required
    default="train",  # Optional: Set a default value
    help="The data subset to process. Must be one of: train, test, or val.",
)

args = parser.parse_args()


logger = MyLogger(
    label="Data Cleanness",
    section_name="Data Cleanness",
    file_log_path=LOG_DIR / "data_cleanness.log",
)

subset: SubsetType = args.subset

input_versioner = VersionedFileManager(
    # file_path=EXTENDED_DATA_DIR / f"{subset}_extended.feather",
    file_path=ENGINEERED_DATA_DIR / f"{subset}_engineered.feather",
    logger=logger,
)
input_df = load_df(df_file_path=input_versioner.current_newest, logger=logger)


# ---------------------------
# Helpers
# ---------------------------


def log_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_rows(
    df: pd.DataFrame,
    mask: pd.Series,
    title: str,
    cols: list[str] | None = None,
    n: int = 10,
):
    bad = df.loc[mask]
    print(f"\n[{title}] Count: {len(bad)}")
    if len(bad) == 0:
        print("No rows found.")
        return
    if cols is None:
        cols = [
            c
            for c in df.columns
            if c
            in {
                "repo",
                "commit",
                "label",
                "has_bug",
                "message",
                "loc_added",
                "loc_deleted",
                "files_changed",
                "hunks_count",
            }
        ]
    print(bad[cols].head(n).to_string(index=False))


def is_list_like(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def normalize_lines(value: Any) -> list:
    """
    Normalize `lines` field into a Python list if possible.
    Expected examples:
    - []
    - [1, 5, 8]
    - np.array([...])
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def lines_are_valid(value: Any) -> bool:
    vals = normalize_lines(value)
    if not isinstance(vals, list):
        return False
    for x in vals:
        if not isinstance(x, (int, np.integer)):
            return False
        if x <= 0:
            return False
    return True


def count_lines(value: Any) -> int:
    return len(normalize_lines(value))


def embedding_length(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, np.ndarray):
        return int(value.shape[0]) if value.ndim == 1 else None
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def embedding_is_numeric_1d(value: Any) -> bool:
    if value is None:
        return False
    arr = None
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return False
    else:
        return False

    return arr.ndim == 1 and np.isfinite(arr).all()


def safe_str_len(x: Any) -> int:
    return len(x) if isinstance(x, str) else 0


def is_probable_commit_hash(x: Any) -> bool:
    if not isinstance(x, str):
        return False
    x = x.strip()
    if len(x) not in {7, 8, 10, 12, 16, 20, 24, 32, 40}:
        return False
    allowed = set("0123456789abcdef")
    return all(ch in allowed for ch in x.lower())


def iqr_outlier_mask(series: pd.Series, factor: float = 3.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (s < lower) | (s > upper)


def normalize_nested_lines(value):
    """
    Expected structure after commit-level grouping:
    [
        [18, 59, 64],   # file 1 buggy lines
        [],             # file 2 no buggy lines
        [3, 7, 8]       # file 3 buggy lines
    ]
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        return []

    normalized = []
    for inner in value:
        if isinstance(inner, np.ndarray):
            inner = inner.tolist()
        elif inner is None:
            inner = []
        elif not isinstance(inner, list):
            return []  # malformed structure
        normalized.append(inner)

    return normalized


def nested_lines_are_valid(value) -> bool:
    outer = normalize_nested_lines(value)
    if not isinstance(outer, list):
        return False

    for inner in outer:
        if not isinstance(inner, list):
            return False
        for x in inner:
            if not isinstance(x, (int, np.integer)):
                return False
            if x <= 0:
                return False
    return True


def count_file_lists(value) -> int:
    outer = normalize_nested_lines(value)
    return len(outer)


def count_total_bug_lines(value) -> int:
    outer = normalize_nested_lines(value)
    return sum(len(inner) for inner in outer)


def count_buggy_files(value) -> int:
    outer = normalize_nested_lines(value)
    return sum(1 for inner in outer if len(inner) > 0)


# ---------------------------
# Initial overview
# ---------------------------

log_header("BASIC INFO")
print(input_df.info())
print("\nShape:", input_df.shape)
print("\nColumns:", list(input_df.columns))

expected_columns = [
    "repo",
    "commit",
    "lines",
    "content",
    "loc_added",
    "loc_deleted",
    "files_changed",
    "hunks_count",
    "msg_len",
    "has_fix_kw",
    "has_bug_kw",
    "author_exp_pre",
    "author_recent_activity_pre",
    "recent_churn",
    "time_since_last_change",
    "ast_delta",
    "max_func_change",
    "complexity_delta",
    "todo",
    "fixme",
    "try",
    "except",
    "raise",
    "code_embed",
    "msg_embed",
    "has_bug",
    "label",
    "message",
]

log_header("SCHEMA CHECK")
missing_cols = sorted(set(expected_columns) - set(input_df.columns))
extra_cols = sorted(set(input_df.columns) - set(expected_columns))

print("Missing expected columns:", missing_cols if missing_cols else "None")
print("Unexpected extra columns:", extra_cols if extra_cols else "None")


# ---------------------------
# Missing values
# ---------------------------

log_header("MISSING VALUES")
missing_counts = input_df.isna().sum().sort_values(ascending=False)
print(missing_counts.to_string())

any_missing = missing_counts[missing_counts > 0]
if len(any_missing) == 0:
    print("\nNo missing values detected.")


# ---------------------------
# Duplicate checks
# ---------------------------

log_header("DUPLICATE CHECKS")

# dup_all = input_df.duplicated().sum()
non_hashable_cols = ["code_embed", "msg_embed", "lines", "filepath"]
dup_subset_cols = [c for c in input_df.columns if c not in non_hashable_cols]
dup_all = input_df.duplicated(subset=dup_subset_cols).sum()
dup_commit_global = input_df.duplicated(subset=["commit"]).sum()
dup_repo_commit = input_df.duplicated(subset=["repo", "commit"]).sum()

print(f"Exact duplicate rows: {dup_all}")
print(f"Duplicate commit hashes globally: {dup_commit_global}")
print(f"Duplicate (repo, commit) pairs: {dup_repo_commit}")

if dup_repo_commit > 0:
    mask_dup_repo_commit = input_df.duplicated(subset=["repo", "commit"], keep=False)
    show_rows(
        input_df,
        mask_dup_repo_commit,
        "Duplicate (repo, commit) rows",
        cols=["repo", "commit", "label", "has_bug", "loc_added", "loc_deleted"],
        n=20,
    )


# ---------------------------
# Basic text/hash sanity
# ---------------------------

log_header("TEXT / HASH SANITY")

mask_repo_empty = input_df["repo"].astype(str).str.strip().eq("")
mask_commit_empty = input_df["commit"].astype(str).str.strip().eq("")
mask_message_empty = input_df["message"].astype(str).str.strip().eq("")
mask_content_empty = input_df["content"].astype(str).str.strip().eq("")

show_rows(input_df, mask_repo_empty, "Empty repo", ["repo", "commit"])
show_rows(input_df, mask_commit_empty, "Empty commit", ["repo", "commit"])
show_rows(input_df, mask_message_empty, "Empty message", ["repo", "commit", "message"])
show_rows(input_df, mask_content_empty, "Empty content", ["repo", "commit", "content"])

mask_commit_bad = ~input_df["commit"].map(is_probable_commit_hash)
show_rows(
    input_df, mask_commit_bad, "Suspicious commit hash format", ["repo", "commit"]
)


# ---------------------------
# Label consistency
# ---------------------------

log_header("LABEL CONSISTENCY")

print("label value counts:")
print(input_df["label"].value_counts(dropna=False).sort_index().to_string())

print("\nhas_bug value counts:")
print(input_df["has_bug"].value_counts(dropna=False).to_string())

mask_label_not_binary = ~input_df["label"].isin([0, 1])
mask_has_bug_not_bool = ~input_df["has_bug"].map(
    lambda x: isinstance(x, (bool, np.bool_))
)
mask_label_bug_mismatch = input_df["label"].astype(int) != input_df["has_bug"].astype(
    int
)

show_rows(
    input_df, mask_label_not_binary, "Non-binary label", ["repo", "commit", "label"]
)
show_rows(
    input_df,
    mask_has_bug_not_bool,
    "has_bug not boolean",
    ["repo", "commit", "has_bug"],
)
show_rows(
    input_df,
    mask_label_bug_mismatch,
    "Mismatch between label and has_bug",
    ["repo", "commit", "label", "has_bug"],
)

print(f"\nPositive rate (label=1): {input_df['label'].mean():.4f}")


# ---------------------------
# lines column checks
# ---------------------------

# log_header("LINES COLUMN CHECKS")

# line_counts = input_df["lines"].map(count_lines)
# mask_lines_invalid = ~input_df["lines"].map(lines_are_valid)

# print("Line count distribution:")
# print(line_counts.describe().to_string())

# show_rows(input_df, mask_lines_invalid, "Invalid lines format", ["repo", "commit", "lines", "label"])

# # Optional consistency rule:
# # if label=1, often lines should contain at least one buggy line
# # if label=0, usually lines should be empty
# mask_label1_empty_lines = (input_df["label"] == 1) & (line_counts == 0)
# mask_label0_nonempty_lines = (input_df["label"] == 0) & (line_counts > 0)

# show_rows(
#     input_df,
#     mask_label1_empty_lines,
#     "label=1 but lines empty",
#     ["repo", "commit", "label", "lines"],
# )
# show_rows(
#     input_df,
#     mask_label0_nonempty_lines,
#     "label=0 but lines non-empty",
#     ["repo", "commit", "label", "lines"],
# )

# print(f"\nRows with label=1 and empty lines: {mask_label1_empty_lines.sum()}")
# print(f"Rows with label=0 and non-empty lines: {mask_label0_nonempty_lines.sum()}")

log_header("LINES COLUMN CHECKS")

file_list_counts = input_df["lines"].map(count_file_lists)
total_bug_line_counts = input_df["lines"].map(count_total_bug_lines)
buggy_file_counts = input_df["lines"].map(count_buggy_files)

mask_lines_invalid = ~input_df["lines"].map(nested_lines_are_valid)

print("Number of per-file line lists per commit:")
print(file_list_counts.describe().to_string())

print("\nTotal number of buggy lines per commit:")
print(total_bug_line_counts.describe().to_string())

print("\nNumber of buggy files per commit:")
print(buggy_file_counts.describe().to_string())

show_rows(
    input_df,
    mask_lines_invalid,
    "Invalid nested lines format",
    ["repo", "commit", "lines", "label"],
)

mask_label1_no_bug_lines = (input_df["label"] == 1) & (total_bug_line_counts == 0)
mask_label0_with_bug_lines = (input_df["label"] == 0) & (total_bug_line_counts > 0)

show_rows(
    input_df,
    mask_label1_no_bug_lines,
    "label=1 but no actual buggy lines present",
    ["repo", "commit", "label", "lines"],
)

show_rows(
    input_df,
    mask_label0_with_bug_lines,
    "label=0 but actual buggy lines present",
    ["repo", "commit", "label", "lines"],
)

print(
    f"\nRows with label=1 but no actual buggy lines: {mask_label1_no_bug_lines.sum()}"
)
print(
    f"Rows with label=0 but actual buggy lines present: {mask_label0_with_bug_lines.sum()}"
)

# ---------------------------
# Numeric range and sanity checks
# ---------------------------

log_header("NUMERIC SANITY CHECKS")

numeric_cols = [
    "loc_added",
    "loc_deleted",
    "files_changed",
    "hunks_count",
    "msg_len",
    "has_fix_kw",
    "has_bug_kw",
    "author_exp_pre",
    "author_recent_activity_pre",
    "recent_churn",
    "time_since_last_change",
    "ast_delta",
    "max_func_change",
    "complexity_delta",
    "todo",
    "fixme",
    "try",
    "except",
    "raise",
    "label",
]

print(input_df[numeric_cols].describe().T.to_string())

non_negative_cols = [
    "loc_added",
    "loc_deleted",
    "files_changed",
    "hunks_count",
    "msg_len",
    "has_fix_kw",
    "has_bug_kw",
    "author_exp_pre",
    "author_recent_activity_pre",
    "recent_churn",
    "time_since_last_change",
    "ast_delta",
    "max_func_change",
    "todo",
    "fixme",
    "try",
    "except",
    "raise",
]

for col in non_negative_cols:
    mask_neg = pd.to_numeric(input_df[col], errors="coerce") < 0
    show_rows(input_df, mask_neg, f"Negative values in {col}", ["repo", "commit", col])

# complexity_delta may legitimately be negative, so don't flag negatives there


# suspicious combinations
mask_zero_files_but_churn = (input_df["files_changed"] == 0) & (
    (input_df["loc_added"] > 0)
    | (input_df["loc_deleted"] > 0)
    | (input_df["hunks_count"] > 0)
)

mask_nonzero_files_but_zero_everything = (
    (input_df["files_changed"] > 0)
    & (input_df["loc_added"] == 0)
    & (input_df["loc_deleted"] == 0)
    & (input_df["hunks_count"] == 0)
)

mask_msg_len_mismatch = input_df["msg_len"] != input_df["message"].map(safe_str_len)
mask_kw_bug_inconsistency = (input_df["has_bug_kw"] > 0) & ~input_df[
    "message"
].str.lower().str.contains("bug", na=False)
mask_kw_fix_inconsistency = (input_df["has_fix_kw"] > 0) & ~input_df[
    "message"
].str.lower().str.contains("fix", na=False)

show_rows(
    input_df,
    mask_zero_files_but_churn,
    "files_changed=0 but churn/hunks nonzero",
    ["repo", "commit", "files_changed", "loc_added", "loc_deleted", "hunks_count"],
)
show_rows(
    input_df,
    mask_nonzero_files_but_zero_everything,
    "files_changed>0 but loc/hunks all zero",
    ["repo", "commit", "files_changed", "loc_added", "loc_deleted", "hunks_count"],
)
show_rows(
    input_df,
    mask_msg_len_mismatch,
    "msg_len does not match actual message length",
    ["repo", "commit", "msg_len", "message"],
)
show_rows(
    input_df,
    mask_kw_bug_inconsistency,
    "has_bug_kw>0 but 'bug' not found in message",
    ["repo", "commit", "has_bug_kw", "message"],
)
show_rows(
    input_df,
    mask_kw_fix_inconsistency,
    "has_fix_kw>0 but 'fix' not found in message",
    ["repo", "commit", "has_fix_kw", "message"],
)


# ---------------------------
# Embedding checks
# ---------------------------

log_header("EMBEDDING CHECKS")

code_embed_len = input_df["code_embed"].map(embedding_length)
msg_embed_len = input_df["msg_embed"].map(embedding_length)

print("code_embed length stats:")
print(code_embed_len.describe().to_string())

print("\nmsg_embed length stats:")
print(msg_embed_len.describe().to_string())

mask_code_embed_invalid = ~input_df["code_embed"].map(embedding_is_numeric_1d)
mask_msg_embed_invalid = ~input_df["msg_embed"].map(embedding_is_numeric_1d)

show_rows(
    input_df,
    mask_code_embed_invalid,
    "Invalid code embeddings",
    ["repo", "commit", "code_embed"],
)
show_rows(
    input_df,
    mask_msg_embed_invalid,
    "Invalid message embeddings",
    ["repo", "commit", "msg_embed"],
)

if code_embed_len.notna().any():
    mode_code_len = code_embed_len.mode(dropna=True)
    common_code_len = mode_code_len.iloc[0] if len(mode_code_len) else None
    mask_code_embed_odd_len = code_embed_len != common_code_len
    show_rows(
        input_df,
        mask_code_embed_odd_len,
        f"code_embed length differs from common length={common_code_len}",
        ["repo", "commit", "code_embed"],
    )

if msg_embed_len.notna().any():
    mode_msg_len = msg_embed_len.mode(dropna=True)
    common_msg_len = mode_msg_len.iloc[0] if len(mode_msg_len) else None
    mask_msg_embed_odd_len = msg_embed_len != common_msg_len
    show_rows(
        input_df,
        mask_msg_embed_odd_len,
        f"msg_embed length differs from common length={common_msg_len}",
        ["repo", "commit", "msg_embed"],
    )


# ---------------------------
# Outlier checks
# ---------------------------

log_header("OUTLIER CHECKS")

outlier_cols = [
    "loc_added",
    "loc_deleted",
    "files_changed",
    "hunks_count",
    "msg_len",
    "author_exp_pre",
    "author_recent_activity_pre",
    "recent_churn",
    "time_since_last_change",
    "ast_delta",
    "max_func_change",
    "complexity_delta",
]

for col in outlier_cols:
    mask_out = iqr_outlier_mask(input_df[col], factor=5.0)
    count_out = int(mask_out.sum())
    print(f"{col}: {count_out} extreme outliers")
    if count_out > 0:
        show_rows(
            input_df,
            mask_out,
            f"Extreme outliers in {col}",
            ["repo", "commit", col, "label"],
            n=5,
        )


# ---------------------------
# Correlation sanity check
# ---------------------------

# -----------------------------------------------------------------------------
# PREPROCESSING: Handling high correlation - Capping high outliers and logging high-correlated features
# -----------------------------------------------------------------------------
input_df = input_df[input_df["loc_added"] < input_df["loc_added"].quantile(0.99)]
input_df = input_df[input_df["ast_delta"] < input_df["ast_delta"].quantile(0.99)]
input_df = input_df[
    input_df["complexity_delta"] < input_df["complexity_delta"].quantile(0.99)
]
# input_df = input_df[input_df["files_changed"] < input_df["files_changed"].quantile(0.99)]
# input_df = input_df[input_df["hunks_count"] < input_df["hunks_count"].quantile(0.99)]
for col in ["loc_added", "ast_delta", "complexity_delta"]:
    input_df[f"{col}"] = np.log1p(input_df[col])


log_header("LIGHT CONSISTENCY SIGNALS")

print("Correlation among a few key size features:")
corr_cols = [
    "loc_added",
    "loc_deleted",
    "files_changed",
    "hunks_count",
    "ast_delta",
    "max_func_change",
    "complexity_delta",
    "author_exp_pre",
]
print(input_df[corr_cols].corr(numeric_only=True).round(3).to_string())

print("\nMean numeric features by label:")
print(
    input_df.groupby("label")[corr_cols + ["msg_len", "recent_churn"]]
    .mean()
    .round(3)
    .to_string()
)


# ---------------------------
# Final summary
# ---------------------------

log_header("FINAL AUDIT SUMMARY")

summary = {
    "rows": len(input_df),
    "exact_duplicate_rows": int(dup_all),
    "duplicate_repo_commit": int(dup_repo_commit),
    "non_binary_label": int(mask_label_not_binary.sum()),
    "label_has_bug_mismatch": int(mask_label_bug_mismatch.sum()),
    "invalid_lines": int(mask_lines_invalid.sum()),
    "label1_empty_lines": int(mask_label1_no_bug_lines.sum()),
    "label0_nonempty_lines": int(mask_label0_with_bug_lines.sum()),
    "zero_files_but_churn": int(mask_zero_files_but_churn.sum()),
    "nonzero_files_but_zero_all_change_metrics": int(
        mask_nonzero_files_but_zero_everything.sum()
    ),
    "msg_len_mismatch": int(mask_msg_len_mismatch.sum()),
    "invalid_code_embed": int(mask_code_embed_invalid.sum()),
    "invalid_msg_embed": int(mask_msg_embed_invalid.sum()),
    "bad_commit_hash_format": int(mask_commit_bad.sum()),
}

for k, v in summary.items():
    print(f"{k}: {v}")

problem_total = sum(v for k, v in summary.items() if k != "rows")
print(f"\nTotal flagged issues across checks: {problem_total}")

if problem_total == 0:
    print("\nDataset looks structurally clean according to these checks.")
else:
    print("\nDataset contains flagged rows. Inspect the printed samples above.")


print(
    input_df[["recent_churn", "repo", "commit", "lines", "hunks_count"]]
    .sort_values("recent_churn", ascending=False)
    .head()
)

input_df["ratio"] = input_df["ast_delta"] / (input_df["loc_added"] + 1)
print(input_df["ratio"].describe())

print(
    input_df.sort_values("loc_added", ascending=False)[
        ["loc_added", "ast_delta", "complexity_delta", "files_changed"]
    ].head(10)
)
