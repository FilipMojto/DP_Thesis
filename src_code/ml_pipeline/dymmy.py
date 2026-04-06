

# from notebooks.logging_config import MyLogger
# from src_code.config import ENGINEERED_DATA_DIR, LOG_DIR
# from src_code.ml_pipeline.preprocessing.feature_config import DROP_COLS
# from src_code.ml_pipeline.preprocessing.preprocessing import drop_cols
# from src_code.versioning import VersionedFileManager
# import src_code.ml_pipeline.data_utils as dutls

# DEF_SCRIPT_LOGGER = MyLogger(
#     label="EVAL_DEPLOY",
#     section_name="EVAL & DEPLOY LOGGER SCRIPT",
#     file_log_path=LOG_DIR / "eval_deploy_log.log",
# )


# if __name__ == "__main__":
#     logger = DEF_SCRIPT_LOGGER
#     test_df_versioner = VersionedFileManager(
#         # file_path=PROCESSED_DATA_DIR / "test_transformed.feather", logger=logger
#         file_path=ENGINEERED_DATA_DIR / "test_engineered.feather",
#         logger=logger,
#     )
#     test_df = dutls.load_df(
#         df_file_path=test_df_versioner.current_newest, logger=logger
#     )

#     test_df = drop_cols(df=test_df, cols=DROP_COLS, logger=logger)

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
import numpy as np
from sklearn.metrics import accuracy_score

from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERED_DATA_DIR, LOG_DIR
from src_code.ml_pipeline.preprocessing.feature_config import DROP_COLS
from src_code.ml_pipeline.preprocessing.preprocessing import drop_cols
from src_code.versioning import VersionedFileManager
import src_code.ml_pipeline.data_utils as dutls

# =============================================================================
# LOGGER
# =============================================================================

logger = MyLogger(
    label="BASELINE_EVAL",
    section_name="BASELINE EVALUATION",
    file_log_path=LOG_DIR / "baseline_eval.log",
)

# =============================================================================
# THRESHOLD OPTIMIZATION (F2)
# =============================================================================

def find_best_f2_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-8)

    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0

    return best_threshold, f2_scores[best_idx]


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_baseline(model, X, y, name="baseline"):
    logger.log_check(f"Evaluating {name}")

    # Fit model
    model.fit(X, y)

    # Predictions
    preds = model.predict(X)

    # Probabilities (important: works for DummyClassifier)
    probs = model.predict_proba(X)[:, 1]

    # Metrics
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f2 = fbeta_score(y, preds, beta=2, zero_division=0)

    roc_auc = roc_auc_score(y, probs)
    auprc = average_precision_score(y, probs)

    # Threshold optimization
    best_thresh, best_f2 = find_best_f2_threshold(y, probs)
    preds_opt = (probs >= best_thresh).astype(int)

    precision_opt = precision_score(y, preds_opt, zero_division=0)
    recall_opt = recall_score(y, preds_opt, zero_division=0)

    accuracy = accuracy_score(y, preds)
    accuracy_opt = accuracy_score(y, preds_opt)

    # Logging
    logger.log_result(f"\n=== {name} ===")
    logger.log_result(f"Default Accuracy: {accuracy:.4f}")
    logger.log_result(f"Optimized Accuracy: {accuracy_opt:.4f}")

    logger.log_result(f"Default Precision: {precision:.4f}")
    logger.log_result(f"Default Recall: {recall:.4f}")
    logger.log_result(f"Default F2: {f2:.4f}")

    logger.log_result(f"ROC-AUC: {roc_auc:.4f}")
    logger.log_result(f"AUPRC: {auprc:.4f}")

    logger.log_result(f"Best Threshold (F2): {best_thresh:.4f}")
    logger.log_result(f"Best F2: {best_f2:.4f}")
    logger.log_result(f"Precision@Best: {precision_opt:.4f}")
    logger.log_result(f"Recall@Best: {recall_opt:.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Load data
    test_df_versioner = VersionedFileManager(
        file_path=ENGINEERED_DATA_DIR / "test_engineered.feather",
        logger=logger,
    )

    df = dutls.load_df(
        df_file_path=test_df_versioner.current_newest,
        logger=logger,
    )

    df = drop_cols(df=df, cols=DROP_COLS, logger=logger)

    X = df.drop(columns=["label"])
    y = df["label"]

    # =============================================================================
    # BASELINES
    # =============================================================================

    random_clf = DummyClassifier(strategy="stratified", random_state=42)
    majority_clf = DummyClassifier(strategy="most_frequent")

    evaluate_baseline(random_clf, X, y, name="Random (Stratified)")
    evaluate_baseline(majority_clf, X, y, name="Majority Classifier")