from dataclasses import dataclass

import numpy as np

@dataclass
class EvaluationResult:
    model_name: str
    y_true: np.ndarray
    probs: np.ndarray
    preds_default: np.ndarray
    preds_thresholded: np.ndarray
    pr_curve: tuple  # (precision, recall, thresholds)
    roc_curve: tuple # (fpr, tpr)
    roc_auc: float
    mcc_threshold: float
    classification_report: dict