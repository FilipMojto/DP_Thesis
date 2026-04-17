from __future__ import annotations

from typing import Dict, Tuple, List, Optional, get_args, get_args

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERED_DATA_DIR, EXTRACTED_DFS, LOG_DIR, SubsetType
from src_code.ml_pipeline.data_utils import load_df
from src_code.versioning import VersionedFileManager


class DropAllMissingAndConstant(BaseEstimator, TransformerMixin):
    """
    Drop columns that are entirely missing or constant on the fitted training fold.
    Works with pandas DataFrames.
    """

    def __init__(self):
        self.keep_columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        keep = []
        for col in X.columns:
            series = X[col]
            if series.isna().all():
                continue
            if series.nunique(dropna=False) <= 1:
                continue
            keep.append(col)

        self.keep_columns_ = keep
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.loc[:, self.keep_columns_].copy()


def _pick_base_df(input_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Prefer a train split if present; otherwise use the first dataframe.
    """
    for key in ("train", "training", "tr", "train_df"):
        if key in input_dfs:
            return input_dfs[key]
    return next(iter(input_dfs.values()))

def clean_dataset(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # 1. DROP IDENTIFIERS / RAW SOURCES
    # -----------------------------
    drop_cols = [
        "datetime",
        "commit",
        "repo",
        "filepath",
        "content",
        "methods",
        "lines",
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # -----------------------------
    # 2. DROP FULLY MISSING COLUMNS
    # -----------------------------
    all_nan_cols = df.columns[df.isna().all()]
    df = df.drop(columns=all_nan_cols)

    # -----------------------------
    # 3. DROP AST / BROKEN FEATURES (object + empty)
    # -----------------------------
    ast_like_cols = [
        "ast_node_delta",
        "function_def_delta",
        "class_def_delta",
        "cyclomatic_complexity_delta",
        "ast_node_count_before",
        "ast_node_count_after",
    ]

    df = df.drop(columns=[c for c in ast_like_cols if c in df.columns])

    # -----------------------------
    # 4. CLEAN INF VALUES (numeric only)
    # -----------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    return df

# def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
#     """
#     Build preprocessing for numeric and categorical columns.
#     """
#     numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
#     categorical_cols = X.columns.difference(numeric_cols).tolist()

#     numeric_pipe = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#         ]
#     )

#     categorical_pipe = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
#         ]
#     )

#     return ColumnTransformer(
#         transformers=[
#             ("num", numeric_pipe, numeric_cols),
#             ("cat", categorical_pipe, categorical_cols),
#         ],
#         remainder="drop",
#         verbose_feature_names_out=False,
#     )
def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def _build_model(y_train: pd.Series) -> XGBClassifier:
    """
    Create XGBoost with a basic imbalance adjustment.
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def _plot_roc_pr(y_true, y_score, title_prefix: str = "") -> None:
    """
    Plot ROC and PR curves.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_title(f"{title_prefix} ROC Curve (AUC = {roc_auc:.4f})")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    axes[1].plot(rec, prec)
    axes[1].set_title(f"{title_prefix} PR Curve (AUC = {pr_auc:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    plt.tight_layout()
    plt.show()


def _evaluate_at_threshold(y_true, y_prob, threshold: float = 0.5) -> None:
    """
    Print confusion matrix and classification report at a chosen threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n=== Evaluation at threshold = {threshold:.2f} ===")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))


def safe_replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        # only process numeric-like columns
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        else:
            print(f"Warning: Column '{col}' is not numeric, skipping inf replacement.")

    return df

# def train_xgb_pipeline(
#     df_train: pd.DataFrame,
#     df_test: pd.DataFrame,
#     target_col: str = "target",
#     drop_cols: Optional[List[str]] = None,
#     test_size: float = 0.2,
#     n_splits: int = 5,
# ) -> None:
#     """
#     Train XGBoost with leakage-safe preprocessing, stratified CV, and hold-out evaluation.
#     """
#     drop_cols = drop_cols or []
#     df_train = df_train.copy()

#     # Basic cleanup
#     # df = df.replace([np.inf, -np.inf], np.nan)
#     df_train = clean_dataset(df_train, target_col=target_col)
#     df_train = safe_replace_inf(df_train)

#     # Remove leakage / identifier columns if present
#     safe_drop_cols = [c for c in drop_cols if c in df_train.columns]
#     if safe_drop_cols:
#         df_train = df_train.drop(columns=safe_drop_cols)

#     if target_col not in df_train.columns:
#         raise ValueError(f"Target column '{target_col}' not found.")

#     y = df_train[target_col].astype(int)
#     X = df_train.drop(columns=[target_col])

#     # Hold-out split first
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=test_size,
#         stratify=y,
#         random_state=42,
#     )

#     # Pipeline: drop bad columns inside CV, then preprocess, then train model
#     preprocessor = build_preprocessor(X_train)
#     model = _build_model(y_train)

#     pipeline = Pipeline(
#         steps=[
#             ("drop_bad", DropAllMissingAndConstant()),
#             ("prep", preprocessor),
#             ("model", model),
#         ]
#     )

#     # Stratified CV on training split only
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     cv_roc = []
#     cv_pr = []

#     print("\n=== Cross-validation on training split ===")
#     for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
#         X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
#         y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

#         fold_pipe = Pipeline(
#             steps=[
#                 ("drop_bad", DropAllMissingAndConstant()),
#                 ("prep", build_preprocessor(X_tr)),
#                 ("model", _build_model(y_tr)),
#             ]
#         )

#         fold_pipe.fit(X_tr, y_tr)
#         y_val_prob = fold_pipe.predict_proba(X_val)[:, 1]

#         roc = roc_auc_score(y_val, y_val_prob)
#         pr = average_precision_score(y_val, y_val_prob)

#         cv_roc.append(roc)
#         cv_pr.append(pr)

#         print(f"Fold {fold}: ROC-AUC = {roc:.4f}, PR-AUC = {pr:.4f}")

#     print("\n=== CV summary ===")
#     print(f"ROC-AUC: {np.mean(cv_roc):.4f} ± {np.std(cv_roc):.4f}")
#     print(f"PR-AUC : {np.mean(cv_pr):.4f} ± {np.std(cv_pr):.4f}")

#     # Fit final model on full training split
#     pipeline.fit(X_train, y_train)

#     # Evaluate on hold-out test split
#     # y_test_prob = pipeline.predict_proba(X_test)[:, 1]

#     # Evaluate on test set
#     y_test_prob = pipeline.predict_proba(X_test)[:, 1]

#     roc_test = roc_auc_score(y_test, y_test_prob)
#     pr_test = average_precision_score(y_test, y_test_prob)

#     print("\n=== Hold-out test evaluation ===")
#     print(f"ROC-AUC: {roc_test:.4f}")
#     print(f"PR-AUC : {pr_test:.4f}")

#     _evaluate_at_threshold(y_test, y_test_prob, threshold=0.5)
#     _plot_roc_pr(y_test, y_test_prob, title_prefix="Test")

def train_xgb_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "target",
    drop_cols: Optional[List[str]] = None,
    n_splits: int = 5,
) -> None:

    drop_cols = drop_cols or []

    # -----------------------------
    # CLEAN TRAIN + TEST
    # -----------------------------
    df_train = clean_dataset(df_train, target_col=target_col)
    df_train = safe_replace_inf(df_train)

    df_test = clean_dataset(df_test, target_col=target_col)
    df_test = safe_replace_inf(df_test)

    # -----------------------------
    # DROP LEAKAGE COLUMNS
    # -----------------------------
    safe_drop_cols = [c for c in drop_cols if c in df_train.columns]

    df_train = df_train.drop(columns=safe_drop_cols)
    df_test = df_test.drop(columns=[c for c in safe_drop_cols if c in df_test.columns])

    # -----------------------------
    # SPLIT FEATURES / TARGET
    # -----------------------------
    y_train = df_train[target_col].astype(int)
    X_train = df_train.drop(columns=[target_col])

    y_test = df_test[target_col].astype(int)
    X_test = df_test.drop(columns=[target_col])

    # -----------------------------
    # CV ON TRAIN ONLY
    # -----------------------------
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_roc, cv_pr = [], []

    print("\n=== Cross-validation on training set ===")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        fold_pipe = Pipeline([
            ("drop_bad", DropAllMissingAndConstant()),
            ("prep", build_preprocessor(X_tr)),
            ("model", _build_model(y_tr)),
        ])

        fold_pipe.fit(X_tr, y_tr)
        val_prob = fold_pipe.predict_proba(X_val)[:, 1]

        roc = roc_auc_score(y_val, val_prob)
        pr = average_precision_score(y_val, val_prob)

        cv_roc.append(roc)
        cv_pr.append(pr)

        print(f"Fold {fold}: ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}")

    print("\n=== CV SUMMARY ===")
    print(f"ROC-AUC: {np.mean(cv_roc):.4f} ± {np.std(cv_roc):.4f}")
    print(f"PR-AUC : {np.mean(cv_pr):.4f} ± {np.std(cv_pr):.4f}")

    # -----------------------------
    # FINAL MODEL (TRAIN ON FULL TRAIN SET)
    # -----------------------------
    final_pipeline = Pipeline([
        ("drop_bad", DropAllMissingAndConstant()),
        ("prep", build_preprocessor(X_train)),
        ("model", _build_model(y_train)),
    ])

    final_pipeline.fit(X_train, y_train)

    # -----------------------------
    # FINAL TEST EVALUATION (TRUE HOLDOUT)
    # -----------------------------
    test_prob = final_pipeline.predict_proba(X_test)[:, 1]

    roc_test = roc_auc_score(y_test, test_prob)
    pr_test = average_precision_score(y_test, test_prob)

    print("\n=== FINAL TEST EVALUATION ===")
    print(f"ROC-AUC: {roc_test:.4f}")
    print(f"PR-AUC : {pr_test:.4f}")

    _evaluate_at_threshold(y_test, test_prob, threshold=0.5)
    _plot_roc_pr(y_test, test_prob, title_prefix="Test")

if __name__ == "__main__":
    logger = MyLogger(
        label="train",
        section_name="train",
        file_log_path=LOG_DIR / "train_v2.log",
    )

    analyzed_subsets = get_args(SubsetType)

    input_dfs: Dict[str, pd.DataFrame] = {}
    for analyzed_subset in analyzed_subsets:
        train_df_versioner = VersionedFileManager(
            file_path=EXTRACTED_DFS / f"{analyzed_subset}_extracted.feather",
            logger=logger,
        )
        input_dfs[analyzed_subset] = load_df(df_file_path=train_df_versioner.current_newest, logger=logger)
        logger.log_result(
            f"Training set {analyzed_subset} loaded with shape: {input_dfs[analyzed_subset].shape}"
        )

    df = _pick_base_df(input_dfs)

    # Drop obvious identifier columns; keep if you explicitly want them
    leakage_like_cols = ["commit", "repo", "filepath"]

    train_xgb_pipeline(
        df_train=input_dfs['train'],
        df_test=input_dfs['test'],
        target_col="target",
        drop_cols=leakage_like_cols,
        # test_size=0.2,
        n_splits=5,
    )