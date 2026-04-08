import math
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from notebooks.logging_config import MyLogger
from src_code.config import SupportedModel
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.experimenting.types import EvalResults
from src_code.ml_pipeline.testing.objects import EvaluationResult


def infer(model: BaseEstimator, X_test, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Performing final model inference...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[
        :, 1
    ]  # Probability of the positive class
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
    model_name: SupportedModel,
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

    best_thresh, best_f2_score = find_optimal_threshold_F2(
        precision, recall, pr_thresholds, logger
    )

    preds_thresh = (probs >= best_thresh).astype(int)

    report = classification_report(y_true, preds_thresh, output_dict=True)

    logger.log_result(f"ROC-AUC: {roc_auc:.4f}")

    auprc = average_precision_score(y_true, probs)
    logger.log_result(f"AUPRC: {auprc:.4f}")

    # return EvaluationResult(
    #     model_name=model_name,
    #     y_true=y_true,
    #     probs=probs,
    #     preds_default=preds,
    #     preds_thresholded=preds_thresh,
    #     pr_curve=(precision, recall, pr_thresholds),
    #     roc_curve=(fpr, tpr),
    #     roc_auc=roc_auc,
    #     auprc=auprc,
    #     best_threshold=best_thresh,
    #     best_score=best_f2_score,
    #     classification_report=report,
    # )
    return EvalResults(
        model_name=model_name,
        y_true=y_true,
        probs=probs,
        preds_default=preds,
        preds_thresholded=preds_thresh,
        pr_curve=(precision, recall, pr_thresholds),
        roc_curve=(fpr, tpr),
        roc_auc=roc_auc,
        auprc=auprc,
        best_threshold=best_thresh,
        best_score=best_f2_score,
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


def find_best_threshold(
    precision, recall, thresholds, logger: MyLogger = DEF_NOTEBOOK_LOGGER
):
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


def find_optimal_threshold_F2(
    precision, recall, pr_thresholds, logger: MyLogger = DEF_NOTEBOOK_LOGGER
):
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


# def plot_pr_grid(results, experiment_path: Path = None, cols=2):
#     rows = math.ceil(len(results) / cols)
#     fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
#     axes = np.array(axes).reshape(-1)

#     for ax, res in zip(axes, results):
#         precision, recall, _ = res.pr_curve
#         ax.plot(recall, precision)
#         ax.set_title(res.model_name)
#         ax.set_xlabel("Recall")
#         ax.set_ylabel("Precision")
#         ax.grid(True)

#     if experiment_path:
#         save_file = experiment_path / "precision_recall_curves.png"
#         plt.savefig(save_file)
#         print(f"Saved PR grid to {save_file}")

#     plt.tight_layout()
#     plt.show()

def plot_pr_combined(results: List[EvalResults], experiment_path: Path = None):
    """Newer combined plot for PR curves that can handle EvalResults from experimenting types.
    Old version - plot_pr_grid - is more for quick individual checks, while this one is for final comparisons.
    Args:
        results (List[EvalResults]): _description_
        experiment_path (Path, optional): _description_. Defaults to None.
    """
    plt.figure(figsize=(10, 7))
    
    for res in results:
        # Assuming res.pr_curve returns (precision, recall, thresholds)
        precision, recall, _ = res.pr_curve
        label = f"{res.model_name} (AUPRC: {res.auprc:.3f})"
        plt.plot(recall, precision, label=label, linewidth=2)

    plt.title("Precision-Recall Curves Comparison")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="lower left")
    
    if experiment_path:
        save_file = experiment_path / "precision_recall_combined.png"
        plt.savefig(save_file)
        print(f"Saved PR combined plot to {save_file}")

    plt.show()


# def plot_roc_grid(results, experiment_path: Path = None, cols=2):
#     rows = math.ceil(len(results) / cols)
#     fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
#     axes = np.array(axes).reshape(-1)

#     if experiment_path:
#         save_file = experiment_path / "roc_curves.png"
#         plt.savefig(save_file)
#         print(f"Saved ROC grid to {save_file}")

#     for ax, res in zip(axes, results):
#         fpr, tpr = res.roc_curve
#         ax.plot(fpr, tpr, label=f"AUC={res.roc_auc:.3f}")
#         ax.plot([0, 1], [0, 1], "--", color="gray")
#         ax.set_title(res.model_name)
#         ax.set_xlabel("FPR")
#         ax.set_ylabel("TPR")
#         ax.legend()
#         ax.grid(True)

#     plt.tight_layout()
#     plt.show()

def plot_roc_combined(results: List[EvalResults], experiment_path: Path = None):
    """Newer version of ROC curve plotting that can handle EvalResults from experimenting types.
    Old version - plot_roc_grid - is more for quick individual checks, while this one

    Args:
        results (List[EvalResults]): _description_
        experiment_path (Path, optional): _description_. Defaults to None.
    """
    plt.figure(figsize=(10, 7))
    
    # Plot the diagonal 50/50 line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    for res in results:
        # Assuming res.roc_curve returns (fpr, tpr, thresholds)
        fpr, tpr = res.roc_curve
        label = f"{res.model_name} (AUC: {res.roc_auc:.3f})"
        plt.plot(fpr, tpr, label=label, linewidth=2)

    plt.title("ROC Curves Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="lower right")
    
    if experiment_path:
        save_file = experiment_path / "roc_combined.png"
        plt.savefig(save_file)

    plt.show()