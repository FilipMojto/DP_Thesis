from __future__ import annotations

import argparse
from typing import get_args

from notebooks.logging_config import MyLogger
from src_code.config import EXTRACTED_DFS, LOG_DIR, SubsetType
from src_code.ml_pipeline.data_utils import load_df
from src_code.versioning import VersionedFileManager

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# -----------------------------
# Generic helpers
# -----------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number", "bool"]).columns.tolist()


def _categorical_columns(df: pd.DataFrame, exclude: List[str] | None = None) -> List[str]:
    exclude = exclude or []
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            cols.append(c)
    return cols


# -----------------------------
# Core EDA summaries
# -----------------------------

# def summarize_shape(df: pd.DataFrame, logger) -> Dict[str, object]:
#     info = {
#         "rows": df.shape[0],
#         "cols": df.shape[1],
#         "dup_rows": int(df.duplicated().sum()),
#     }
#     logger.log_result(f"Shape: {df.shape}")
#     logger.log_result(f"Duplicate rows: {info['dup_rows']}")
#     return info
def summarize_shape(df: pd.DataFrame, logger) -> Dict[str, object]:
    dup_subset = [c for c in ["commit", "repo", "filepath"] if c in df.columns]

    if dup_subset:
        dup_rows = int(df.duplicated(subset=dup_subset).sum())
    else:
        dup_rows = int(df.select_dtypes(exclude=["object"]).duplicated().sum())

    info = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "dup_rows": dup_rows,
    }
    logger.log_result(f"Shape: {df.shape}")
    logger.log_result(f"Duplicate rows based on {dup_subset if dup_subset else 'non-object columns'}: {info['dup_rows']}")
    return info


def missing_values_report(df: pd.DataFrame, logger) -> pd.DataFrame:
    miss = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().sum() / len(df)) * 100,
        "dtype": df.dtypes.astype(str),
    }).sort_values("missing_pct", ascending=False)

    logger.log_result("Missing values report:")
    logger.log_result(miss[miss["missing_count"] > 0].to_string())
    return miss


def target_report(df: pd.DataFrame, target_col: str, logger) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    counts = df[target_col].value_counts(dropna=False).sort_index()
    pct = df[target_col].value_counts(normalize=True, dropna=False).sort_index() * 100

    report = pd.DataFrame({
        "count": counts,
        "pct": pct,
    })

    logger.log_result(f"Target distribution for '{target_col}':")
    logger.log_result(report.to_string())
    return report


def numeric_summary(df: pd.DataFrame, logger) -> pd.DataFrame:
    num_cols = _numeric_columns(df)
    if not num_cols:
        logger.log_result("No numeric columns found.")
        return pd.DataFrame()

    summary = df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    logger.log_result("Numeric summary:")
    logger.log_result(summary.to_string())
    return summary


def categorical_summary(df: pd.DataFrame, logger, max_categories: int = 20) -> Dict[str, pd.DataFrame]:
    cat_cols = _categorical_columns(df)
    results = {}

    for col in cat_cols:
        vc = df[col].fillna("<<MISSING>>").astype(str).value_counts(dropna=False).head(max_categories)
        results[col] = vc.to_frame(name="count")
        logger.log_result(f"Top categories for '{col}':")
        logger.log_result(results[col].to_string())

    return results


# -----------------------------
# Outlier detection
# -----------------------------

def iqr_outlier_report(df: pd.DataFrame, numeric_cols: List[str], logger) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_count = 0
            lower = upper = q1
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((s < lower) | (s > upper)).sum())

        rows.append({
            "feature": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": outlier_count,
            "outlier_pct": (outlier_count / len(s)) * 100 if len(s) else 0.0,
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean(),
            "median": s.median(),
            "skew": s.skew(),
        })

    out = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)
    logger.log_result("IQR outlier report:")
    if not out.empty:
        logger.log_result(out.to_string(index=False))
    else:
        logger.log_result("No numeric columns available for outlier analysis.")
    return out


# -----------------------------
# Domain-specific JIT EDA
# -----------------------------

def repo_level_report(df: pd.DataFrame, logger, target_col: str = "target") -> pd.DataFrame:
    if "repo" not in df.columns:
        logger.log_result("No 'repo' column found.")
        return pd.DataFrame()

    rep = (
        df.groupby("repo")
        .agg(
            rows=("repo", "size"),
            defect_rate=(target_col, "mean"),
        )
        .sort_values("rows", ascending=False)
    )
    rep["defect_rate_pct"] = rep["defect_rate"] * 100

    logger.log_result("Repo-level report (top 20 by rows):")
    logger.log_result(rep.head(20).to_string())
    return rep


def time_report(df: pd.DataFrame, logger, target_col: str = "target") -> pd.DataFrame:
    if "datetime" not in df.columns:
        logger.log_result("No 'datetime' column found.")
        return pd.DataFrame()

    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["datetime"], errors="coerce").dt.to_period("M").astype(str)

    monthly = (
        tmp.groupby("month")
        .agg(
            rows=("month", "size"),
            defect_rate=(target_col, "mean"),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly["defect_rate_pct"] = monthly["defect_rate"] * 100

    logger.log_result("Monthly time report:")
    logger.log_result(monthly.head(30).to_string(index=False))
    return monthly


def path_feature_checks(df: pd.DataFrame, logger) -> pd.DataFrame:
    cols = [c for c in ["path_depth", "is_test_file", "is_init_file", "has_src_dir", "has_docs_dir", "has_tests_dir"] if c in df.columns]
    if not cols:
        logger.log_result("No path-derived columns found.")
        return pd.DataFrame()

    summary = df[cols].describe().T
    logger.log_result("Path feature summary:")
    logger.log_result(summary.to_string())
    return summary


# -----------------------------
# Plots
# -----------------------------

def plot_target_distribution(df: pd.DataFrame, target_col: str, out_dir: Path) -> None:
    counts = df[target_col].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    counts.plot(kind="bar")
    plt.title("Target distribution")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    _savefig(out_dir / "target_distribution.png")


def plot_missing_values(missing_df: pd.DataFrame, out_dir: Path) -> None:
    miss = missing_df[missing_df["missing_count"] > 0].sort_values("missing_count", ascending=True)
    if miss.empty:
        return
    plt.figure(figsize=(10, max(4, 0.35 * len(miss))))
    miss["missing_count"].plot(kind="barh")
    plt.title("Missing values by column")
    plt.xlabel("Missing count")
    _savefig(out_dir / "missing_values.png")


def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: List[str], out_dir: Path, max_cols: int = 12) -> None:
    selected = numeric_cols[:max_cols]
    for col in selected:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        plt.figure(figsize=(7, 4))
        plt.hist(s, bins=40)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        _savefig(out_dir / f"hist_{col}.png")


def plot_numeric_boxplots(df: pd.DataFrame, numeric_cols: List[str], out_dir: Path, max_cols: int = 12) -> None:
    selected = numeric_cols[:max_cols]
    data = []
    labels = []
    for col in selected:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        data.append(s)
        labels.append(col)

    if not data:
        return

    plt.figure(figsize=(max(8, len(data) * 0.8), 5))
    plt.boxplot(data, tick_labels=labels, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.title("Boxplots of numeric features")
    _savefig(out_dir / "numeric_boxplots.png")


def plot_top_repos(repo_report: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    if repo_report.empty:
        return
    top = repo_report.head(top_n)
    plt.figure(figsize=(10, 5))
    top["rows"].plot(kind="bar")
    plt.title(f"Top {top_n} repositories by sample count")
    plt.xlabel("repo")
    plt.ylabel("rows")
    plt.xticks(rotation=45, ha="right")
    _savefig(out_dir / "top_repos.png")


def plot_monthly_defect_rate(monthly_df: pd.DataFrame, out_dir: Path) -> None:
    if monthly_df.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_df["month"], monthly_df["defect_rate_pct"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly defect rate")
    plt.xlabel("Month")
    plt.ylabel("Defect rate (%)")
    _savefig(out_dir / "monthly_defect_rate.png")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif


def _cohens_d(x0, x1):
    """Effect size between two groups."""
    x0 = x0.dropna()
    x1 = x1.dropna()
    if len(x0) < 2 or len(x1) < 2:
        return np.nan

    mean0, mean1 = x0.mean(), x1.mean()
    std0, std1 = x0.std(), x1.std()

    pooled_std = np.sqrt((std0**2 + std1**2) / 2)
    if pooled_std == 0:
        return 0.0

    return (mean1 - mean0) / pooled_std


def analyze_feature_signal(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    out_dir: Path,
    logger,
):
    """
    Analyze feature signal vs target using plots + metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for col in features:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")

        x0 = series[df[target_col] == 0]
        x1 = series[df[target_col] == 1]

        # --------------------
        # Metrics
        # --------------------
        mean0, mean1 = x0.mean(), x1.mean()
        median0, median1 = x0.median(), x1.median()

        cohens_d = _cohens_d(x0, x1)

        # Mutual information (needs no NaNs)
        valid = series.notna()
        if valid.sum() > 0:
            mi = mutual_info_classif(
                series[valid].values.reshape(-1, 1),
                df.loc[valid, target_col],
                discrete_features=False
            )[0]
        else:
            mi = np.nan

        results.append({
            "feature": col,
            "mean_0": mean0,
            "mean_1": mean1,
            "median_0": median0,
            "median_1": median1,
            "cohens_d": cohens_d,
            "mutual_info": mi,
        })

        # --------------------
        # Plots
        # --------------------

        # Histogram
        plt.figure()
        plt.hist(x0.dropna(), bins=40, alpha=0.5, label="target=0")
        plt.hist(x1.dropna(), bins=40, alpha=0.5, label="target=1")
        plt.title(f"{col} distribution by target")
        plt.legend()
        plt.savefig(out_dir / f"{col}_hist.png")
        plt.close()

        # Boxplot
        plt.figure()
        plt.boxplot([x0.dropna(), x1.dropna()], tick_labels=["0", "1"])
        plt.title(f"{col} boxplot by target")
        plt.savefig(out_dir / f"{col}_box.png")
        plt.close()

    # --------------------
    # Save results
    # --------------------
    results_df = pd.DataFrame(results).sort_values(
        by="mutual_info", ascending=False
    )

    results_df.to_csv(out_dir / "feature_signal_summary.csv", index=False)

    logger.log_result("Feature signal analysis complete.")
    logger.log_result(results_df.head(15).to_string())

    return results_df


# -----------------------------
# Main EDA entry point
# -----------------------------

def run_eda(df: pd.DataFrame, out_dir: Path, logger, target_col: str = "target") -> Dict[str, pd.DataFrame]:
    """
    Run EDA and save summary tables + plots.

    Returns a dict of useful reports for further inspection.
    """
    _ensure_dir(out_dir)

    reports: Dict[str, pd.DataFrame] = {}

    reports["shape"] = pd.DataFrame([summarize_shape(df, logger)])
    reports["missing"] = missing_values_report(df, logger)
    reports["target"] = target_report(df, target_col, logger)
    reports["numeric_summary"] = numeric_summary(df, logger)
    reports["categorical_summary"] = pd.DataFrame()
    reports["outliers"] = iqr_outlier_report(df, _numeric_columns(df), logger)
    reports["repo_report"] = repo_level_report(df, logger, target_col=target_col)
    reports["monthly_report"] = time_report(df, logger, target_col=target_col)
    reports["path_summary"] = path_feature_checks(df, logger)

    # Save tables
    for name, report in reports.items():
        if isinstance(report, pd.DataFrame) and not report.empty:
            report.to_csv(out_dir / f"{name}.csv", index=True)

    # Plots
    if target_col in df.columns:
        plot_target_distribution(df, target_col, out_dir)
        plot_missing_values(reports["missing"], out_dir)
        plot_numeric_histograms(df, _numeric_columns(df), out_dir)
        plot_numeric_boxplots(df, _numeric_columns(df), out_dir)
        plot_top_repos(reports["repo_report"], out_dir)
        plot_monthly_defect_rate(reports["monthly_report"], out_dir)

        signal_features = [
        "loc_added",
        "loc_deleted",
        "total_changed_loc",
        "hunk_count",
        "change_intensity",
        "changed_methods_count",
        "path_depth",
        "commit_hour",
    ]

    signal_dir = eda_out_dir / "feature_signal"

    signal_results = analyze_feature_signal(
        df=df,
        features=signal_features,
        target_col="target",
        out_dir=signal_dir,
        logger=logger
    )
    reports["feature_signal"] = signal_results


    logger.log_result(f"EDA artifacts saved to: {out_dir}")
    return reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on the extracted data.")
    parser.add_argument(
        "--subset",
        type=str,
        choices=get_args(SubsetType),
        required=True,
        help="Subset to analyze (train, test, validate)"
    )

    # let's load the extracted data and do some EDA to understand it better
    logger = MyLogger(label="eda", section_name="eda", file_log_path=LOG_DIR / "eda.log")
    args = parser.parse_args()
    analyzed_subset: SubsetType = args.subset

    input_df_versioner = VersionedFileManager(
        file_path=EXTRACTED_DFS / f"{analyzed_subset}_extracted.feather", logger=logger
    )
    df = load_df(df_file_path=input_df_versioner.current_newest, logger=logger)

    eda_out_dir = LOG_DIR / "eda_reports" / analyzed_subset
    reports = run_eda(df=df, out_dir=eda_out_dir, logger=logger, target_col="target")