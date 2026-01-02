import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve
from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.testing.objects import EvaluationResult


def infer(model: BaseEstimator, X_test, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Performing final model inference...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of the positive class
    logger.log_result("Inference complete.")
    return predictions, probabilities


# def evaluate(y_true, predictions, probabilities, logger: MyLogger = DEF_NOTEBOOK_LOGGER, threshold: float = None):
#     logger.log_check("Evaluating model inference...")

#     if threshold:
#         predictions = (probabilities >= threshold).astype(int)


#     logger.log_result(f"Classification Report:\n{classification_report(y_true, predictions)}")
#     logger.log_result(f"ROC-AUC Score: {roc_auc_score(y_true, probabilities):.4f}")

#     logger.log_result("Evaluation complete.")

def classification_report_table(results):
    tables = []

    for res in results:
        df = pd.DataFrame(res.classification_report).T
        df["model"] = res.model_name
        tables.append(df)

    return pd.concat(tables)


def evaluate_model(
    model_name: str,
    model: BaseEstimator,
    X_test,
    y_true,
    logger: MyLogger,
):
    logger.log_check(f"Evaluating model: {model_name}")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Curves
    precision, recall, pr_thresholds = precision_recall_curve(y_true, probs)
    fpr, tpr, _ = roc_curve(y_true, probs)

    # Metrics
    roc_auc = roc_auc_score(y_true, probs)
    # best_thresh, best_mcc = find_optimal_threshold_MCC(y_true, probs, logger)

    # beta = 2
    # f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-10)

    # best_idx = np.argmax(f2)
    # best_thresh = pr_thresholds[best_idx]

    # logger.log_result(f"Optimal Threshold for F2: {best_threshold:.4f}")
    # logger.log_result(f"Best F2 Score: {f2[best_idx]:.4f}")

    best_thresh, best_f2 = find_optimal_threshold_F2(precision, recall, pr_thresholds, logger)

    preds_thresh = (probs >= best_thresh).astype(int)

    report = classification_report(
        y_true, preds_thresh, output_dict=True
    )

    
    logger.log_result(f"ROC-AUC: {roc_auc:.4f}")
    # logger.log_result(f"Best MCC: {best_mcc:.4f} @ {best_thresh:.3f}")
    # logger.log_result(f"Best F2: {best_f2:.4f} @ {best_thresh:.3f}")

    return EvaluationResult(
        model_name=model_name,
        y_true=y_true,
        probs=probs,
        preds_default=preds,
        preds_thresholded=preds_thresh,
        pr_curve=(precision, recall, pr_thresholds),
        roc_curve=(fpr, tpr),
        roc_auc=roc_auc,
        mcc_threshold=best_thresh,
        classification_report=report,
    )

# def prec_recall_curve(y_true, probs, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
#     logger.log_check("Plotting precision recall curve...")
    
#     precision, recall, thresholds = precision_recall_curve(y_true, probs)

#     plt.plot(thresholds, precision[:-1], label="Precision")
#     plt.plot(thresholds, recall[:-1], label="Recall")
#     plt.xlabel("Threshold")
#     plt.title("The Precision-Recall Tradeoff")
#     plt.legend()
#     plt.show()

#     logger.log_result("Plotting complete.")

#     return precision, recall, thresholds

def plot_pr_grid(results, cols=2):
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = np.array(axes).reshape(-1)

    for ax, res in zip(axes, results):
        precision, recall, _ = res.pr_curve
        ax.plot(recall, precision)
        ax.set_title(res.model_name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def find_best_threshold(precision, recall, thresholds, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
    # Calculate F1 for every threshold produced by the PR curve
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    logger.log_result(f"Optimal Threshold for F1: {best_threshold:.4f}")
    logger.log_result(f"Best achievable F1-Score: {f1_scores[best_idx]:.4f}")

    return best_threshold, f1_scores[best_idx]


def find_optimal_threshold_MCC(y_true, probs, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
    thresholds = np.linspace(0, 1, 100)
    mcc_scores = [matthews_corrcoef(y_true, probs >= t) for t in thresholds]

    # 3. Find the best one
    best_threshold = thresholds[np.argmax(mcc_scores)]
    best_mcc = max(mcc_scores)

    logger.log_result(f"Optimal Threshold for MCC: {best_threshold:.4f}")
    logger.log_result(f"Best MCC Score: {best_mcc:.4f}")

    return best_threshold, best_mcc


def find_optimal_threshold_F2(precision, recall, pr_thresholds, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
    beta = 2
    f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-10)

    best_idx = np.argmax(f2)
    best_thresh = pr_thresholds[best_idx]
    logger.log_result(f"Optimal Threshold for F2: {best_thresh:.4f}")
    logger.log_result(f"Best F2 Score: {f2[best_idx]:.4f}")

    return best_thresh, f2[best_idx]


# def display_ROC_curve(y_true, probabilities, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
#     logger.log_check("Displaying ROC curve...")
#     fpr, tpr, _ = roc_curve(y_true, probabilities)

#     plt.plot(fpr, tpr, label="Model")
#     plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve")
#     plt.legend()
#     plt.show()

#     logger.log_result("Displayed successfully.")

def plot_roc_grid(results, cols=2):
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = np.array(axes).reshape(-1)

    for ax, res in zip(axes, results):
        fpr, tpr = res.roc_curve
        ax.plot(fpr, tpr, label=f"AUC={res.roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title(res.model_name)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


