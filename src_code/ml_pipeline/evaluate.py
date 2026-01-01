


import argparse
from typing import get_args

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERING_MAPPINGS, LOG_DIR, MODEL_DIR, SupportedModels
import src_code.ml_pipeline.data_utils as dutls
import  src_code.ml_pipeline.testing.testing as test_utils

if __name__ == "__main__":
    script_logger = MyLogger(label="EVAL_DEPLOY", section_name="EVAL & DEPLOY LOGGER SCRIPT", file_log_path=LOG_DIR / "eval_deploy_log.log")
    script_logger.start_session()
    argparser = argparse.ArgumentParser(
        description="Final Evaluation and Deployment Preparation Script"
    )

    argparser.add_argument(
        "--model",
        choices=get_args(SupportedModels),
        default="rf",
        required=False,
        help="Specify which model type to use: 'rf' for Random Forest, 'xgb' for XGBoost.",
    )

    args = argparser.parse_args()
    MODEL_TYPE: SupportedModels = args.model  # "rf" or "xgb"

    model_path = MODEL_DIR / f"{MODEL_TYPE.upper()}_model_script_train.joblib"
    # model_wrapper = dutls.load_model(path=model_path, logger=script_logger)
    # if not filtered_phases or "eval" in filtered_phases:
    # =============================================================================
    # FINAL EVALUATION
    # =============================================================================
    
    # -----------------------------------------------------------------------------
    # Dataset Loading
    # -----------------------------------------------------------------------------
    target_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS['test']["output"]
    test_df = dutls.load_df(df_file_path=target_df_path, logger=script_logger)
    

    # -----------------------------------------------------------------------------
    # Model Loading
    # -----------------------------------------------------------------------------

    model = dutls.load_model(path=model_path, logger=script_logger)
    model_features = model.feature_names_in_

    if isinstance(model, RandomForestClassifier):
        script_logger.log_result("Loaded model is a Random Forest.")
    elif isinstance(model, XGBClassifier):
        script_logger.log_result("Loaded model is an XGBoost Classifier.")
    else:
        script_logger.log_result("Loaded model is of an unknown type.")

    # -----------------------------------------------------------------------------
    # Column Filtering
    # -----------------------------------------------------------------------------

    X_test = test_df[model_features]

    # -----------------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------------
    
    y_true = test_df["label"] if "label" in test_df.columns else None

    predictions, probabilities = test_utils.infer(X_test=X_test, model=model, logger=script_logger)

    # -----------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------
    
    test_utils.evaluate(y_true=y_true, predictions=predictions, probabilities=probabilities, logger=script_logger)

    # -----------------------------------------------------------------------------
    # Precision-Recall Curve
    # -----------------------------------------------------------------------------
    
    precision, recall, thresholds = test_utils.prec_recall_curve(y_true=y_true, probs=probabilities, logger=script_logger)


    # -----------------------------------------------------------------------------
    # Optimal Threshold for MCC
    # -----------------------------------------------------------------------------
    script_logger.log_check("Finding optimal threshold for MCC...")
    best_mcc_threshold, best_mcc = test_utils.find_optimal_threshold_MCC(y_true=y_true, probs=probabilities, logger=script_logger)

    # 4. Generate the final report
    # final_predictions = (probs >= best_threshold).astype(int)
    # print(classification_report(y_true, final_predictions))
    test_utils.evaluate(
        y_true=y_true,
        predictions=predictions,
        probabilities=probabilities,
        threshold=best_mcc_threshold,
        logger=script_logger
    )

    # -----------------------------------------------------------------------------
    # ROC Curve
    # -----------------------------------------------------------------------------
    
    test_utils.display_ROC_curve(y_true=y_true, probabilities=probabilities, logger=script_logger)
