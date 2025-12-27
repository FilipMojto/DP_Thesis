from matplotlib import pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve
from notebooks.constants import TARGET
from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.feature_config import DROP_COLS
from src_code.ml_pipeline.training.utils import drop_cols

# X_test = drop_cols(df=df, cols=DROP_COLS + [TARGET], logger=logger)

# # for feature in X_test:
# #     if feature not in model_features:
# #         X_test = X_test.drop(feature)
# X_test = X_test[model_features]

        
# # 6. Predict
# y_true = df["label"] if "label" in df.columns else None


def infer(model: BaseEstimator, X_test, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Performing final model inference...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of the positive class
    logger.log_result("Inference complete.")
    return predictions, probabilities


def evaluate(y_true, predictions, probabilities, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER, threshold: float = None):
    logger.log_check("Evaluating model inference...")

    if threshold:
        predictions = (probabilities >= threshold).astype(int)


    logger.log_result(classification_report(y_true, predictions))
    logger.log_result(f"ROC-AUC Score: {roc_auc_score(y_true, probabilities):.4f}")

    logger.log_result("Evaluation complete.")


def prec_recall_curve(y_true, probs, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Plotting precision recall curve...")
    
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.title("The Precision-Recall Tradeoff")
    plt.legend()
    plt.show()

    logger.log_result("Plotting complete.")

    return precision, recall, thresholds


def find_best_threshold(precision, recall, thresholds, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    # Calculate F1 for every threshold produced by the PR curve
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    logger.log_result(f"Optimal Threshold for F1: {best_threshold:.4f}")
    logger.log_result(f"Best achievable F1-Score: {f1_scores[best_idx]:.4f}")

    return best_threshold


def find_optimal_threshold_MCC(y_true, probs, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    thresholds = np.linspace(0, 1, 100)
    mcc_scores = [matthews_corrcoef(y_true, probs >= t) for t in thresholds]

    # 3. Find the best one
    best_threshold = thresholds[np.argmax(mcc_scores)]
    best_mcc = max(mcc_scores)

    logger.log_result(f"Optimal Threshold for MCC: {best_threshold:.4f}")
    logger.log_result(f"Best MCC Score: {best_mcc:.4f}")

    return best_threshold, best_mcc


def display_ROC_curve(y_true, probabilities, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Displaying ROC curve...")
    fpr, tpr, _ = roc_curve(y_true, probabilities)

    plt.plot(fpr, tpr, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    logger.log_result("Displayed successfully.")