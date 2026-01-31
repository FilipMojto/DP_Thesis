import argparse
from typing import get_args

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERING_MAPPINGS, LOG_DIR, MODEL_DIR, PROCESSED_DATA_DIR, SupportedModels
from src_code.ml_pipeline.config import SUPPORTED_MODELS
import src_code.ml_pipeline.data_utils as dutls
from src_code.ml_pipeline.experimenting.utils import log_experiment_id
import src_code.ml_pipeline.testing.testing as test_utils
from src_code.utils.utils import timeit
from src_code.versioning import VersionedFileManager


DEF_SCRIPT_LOGGER = MyLogger(
    label="EVAL_DEPLOY",
    section_name="EVAL & DEPLOY LOGGER SCRIPT",
    file_log_path=LOG_DIR / "eval_deploy_log.log",
)

@timeit("Evaluation Phase", logger_name="logger")
def evaluate(
    logger: MyLogger = DEF_SCRIPT_LOGGER,
    models_to_evaluate: list = SUPPORTED_MODELS,
    experiment_id: int = None,
):
    # MODELS = {
    #     "Random Forest": VersionedFileManager(file_path=MODEL_DIR / "RF_model_train.joblib", logger=logger),
    #     "XGBoost": VersionedFileManager(file_path=MODEL_DIR / "XGB_model_train.joblib", logger=logger),
    # }
    log_experiment_id(logger=logger, experiment_id=experiment_id)
    models = {
        model_type: VersionedFileManager(
            file_path=MODEL_DIR / f"{model_type}_model_train.joblib",
            logger=logger,
        )
        for model_type in models_to_evaluate
    }

    results = []


    # =============================================================================
    # FINAL EVALUATION
    # =============================================================================

    # -----------------------------------------------------------------------------
    # Dataset Loading
    # -----------------------------------------------------------------------------
    # target_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS["test"]["output"]
    # test_df = dutls.load_df(df_file_path=target_df_path, logger=script_logger)
    test_df_versioner = VersionedFileManager(file_path=PROCESSED_DATA_DIR / "test_engineered.feather", logger=script_logger)
    test_df = dutls.load_df(df_file_path=test_df_versioner.current_newest, logger=script_logger)
    # # -----------------------------------------------------------------------------
    # # Model Loading
    # # -----------------------------------------------------------------------------

    # model = dutls.load_model(path=model_path, logger=script_logger)
    # model_features = model.feature_names_in_

    # if isinstance(model, RandomForestClassifier):
    #     script_logger.log_result("Loaded model is a Random Forest.")
    # elif isinstance(model, XGBClassifier):
    #     script_logger.log_result("Loaded model is an XGBoost Classifier.")
    # else:
    #     script_logger.log_result("Loaded model is of an unknown type.")

    # # -----------------------------------------------------------------------------
    # # Column Filtering
    # # -----------------------------------------------------------------------------

    # X_test = test_df[model_features]

    # # -----------------------------------------------------------------------------
    # # Inference
    # # -----------------------------------------------------------------------------

    # y_true = test_df["label"] if "label" in test_df.columns else None

    # predictions, probabilities = test_utils.infer(
    #     X_test=X_test, model=model, logger=script_logger
    # )

    # -----------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------

    # test_utils.evaluate(
    #     y_true=y_true,
    #     predictions=predictions,
    #     probabilities=probabilities,
    #     logger=script_logger,
    # )
    for name, verioner in models.items():
        # script_logger.log_check(f"Evaluating model: {name}")
        script_logger.start_section(section_name=f"Evaluating model: {name}")
        model = dutls.load_model(verioner.current_newest, script_logger)
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

        predictions, probabilities = test_utils.infer(
            X_test=X_test, model=model, logger=script_logger
        )

        results.append(
            test_utils.evaluate_model(
                model_name=name,
                model=model,
                X_test=X_test,
                y_true=y_true,
                logger=script_logger,
            )
        )



    report_df = test_utils.classification_report_table(results)
    script_logger.log_result(f"\n{report_df.round(3)}")

    # -----------------------------------------------------------------------------
    # Precision-Recall Curve
    # -----------------------------------------------------------------------------

    # precision, recall, thresholds = test_utils.prec_recall_curve(
    #     y_true=y_true, probs=probabilities, logger=script_logger
    # )

    test_utils.plot_pr_grid(results=results)

    # # -----------------------------------------------------------------------------
    # # Optimal Threshold for MCC
    # # -----------------------------------------------------------------------------
    # script_logger.log_check("Finding optimal threshold for MCC...")
    # best_mcc_threshold, best_mcc = test_utils.find_optimal_threshold_MCC(
    #     y_true=y_true, probs=probabilities, logger=script_logger
    # )

    # 4. Generate the final report
    # final_predictions = (probs >= best_threshold).astype(int)
    # print(classification_report(y_true, final_predictions))
    # test_utils.evaluate(
    #     y_true=y_true,
    #     predictions=predictions,
    #     probabilities=probabilities,
    #     threshold=best_mcc_threshold,
    #     logger=script_logger,
    # )

    # -----------------------------------------------------------------------------
    # ROC Curve
    # -----------------------------------------------------------------------------

    # test_utils.display_ROC_curve(
    #     y_true=y_true, probabilities=probabilities, logger=script_logger
    # )

    test_utils.plot_roc_grid(results=results)



if __name__ == "__main__":
    script_logger = DEF_SCRIPT_LOGGER
    script_logger.start_session()
    # argparser = argparse.ArgumentParser(
    #     description="Final Evaluation and Deployment Preparation Script"
    # )

    # argparser.add_argument(
    #     "--model",
    #     choices=get_args(SupportedModels),
    #     default="rf",
    #     required=False,
    #     help="Specify which model type to use: 'rf' for Random Forest, 'xgb' for XGBoost.",
    # )

    # args = argparser.parse_args()
    # MODEL_TYPE: SupportedModels = args.model  # "rf" or "xgb"

    evaluate(
        logger=script_logger
    )

    # model_file_versioner = VersionedFileManager(file_path=MODEL_DIR / f"{MODEL_TYPE.upper()}_model_train.joblib")

    # MODELS = {
    #     "Random Forest": VersionedFileManager(file_path=MODEL_DIR / "RF_model_train.joblib", logger=script_logger),
    #     "XGBoost": VersionedFileManager(file_path=MODEL_DIR / "XGB_model_train.joblib", logger=script_logger),
    # }

    # results = []


    # # =============================================================================
    # # FINAL EVALUATION
    # # =============================================================================

    # # -----------------------------------------------------------------------------
    # # Dataset Loading
    # # -----------------------------------------------------------------------------
    # target_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS["test"]["output"]
    # # test_df = dutls.load_df(df_file_path=target_df_path, logger=script_logger)
    # test_df_versioner = VersionedFileManager(file_path=PROCESSED_DATA_DIR / "test_engineered.feather", logger=script_logger)
    # test_df = dutls.load_df(df_file_path=test_df_versioner.current_newest, logger=script_logger)
    # # # -----------------------------------------------------------------------------
    # # # Model Loading
    # # # -----------------------------------------------------------------------------

    # # model = dutls.load_model(path=model_path, logger=script_logger)
    # # model_features = model.feature_names_in_

    # # if isinstance(model, RandomForestClassifier):
    # #     script_logger.log_result("Loaded model is a Random Forest.")
    # # elif isinstance(model, XGBClassifier):
    # #     script_logger.log_result("Loaded model is an XGBoost Classifier.")
    # # else:
    # #     script_logger.log_result("Loaded model is of an unknown type.")

    # # # -----------------------------------------------------------------------------
    # # # Column Filtering
    # # # -----------------------------------------------------------------------------

    # # X_test = test_df[model_features]

    # # # -----------------------------------------------------------------------------
    # # # Inference
    # # # -----------------------------------------------------------------------------

    # # y_true = test_df["label"] if "label" in test_df.columns else None

    # # predictions, probabilities = test_utils.infer(
    # #     X_test=X_test, model=model, logger=script_logger
    # # )

    # # -----------------------------------------------------------------------------
    # # Evaluation
    # # -----------------------------------------------------------------------------

    # # test_utils.evaluate(
    # #     y_true=y_true,
    # #     predictions=predictions,
    # #     probabilities=probabilities,
    # #     logger=script_logger,
    # # )
    # for name, verioner in MODELS.items():
    #     # script_logger.log_check(f"Evaluating model: {name}")
    #     script_logger.start_section(section_name=f"Evaluating model: {name}")
    #     model = dutls.load_model(verioner.current_newest, script_logger)
    #     model_features = model.feature_names_in_

    #     if isinstance(model, RandomForestClassifier):
    #         script_logger.log_result("Loaded model is a Random Forest.")
    #     elif isinstance(model, XGBClassifier):
    #         script_logger.log_result("Loaded model is an XGBoost Classifier.")
    #     else:
    #         script_logger.log_result("Loaded model is of an unknown type.")

    #     # -----------------------------------------------------------------------------
    #     # Column Filtering
    #     # -----------------------------------------------------------------------------

    #     X_test = test_df[model_features]

    #     # -----------------------------------------------------------------------------
    #     # Inference
    #     # -----------------------------------------------------------------------------

    #     y_true = test_df["label"] if "label" in test_df.columns else None

    #     predictions, probabilities = test_utils.infer(
    #         X_test=X_test, model=model, logger=script_logger
    #     )

    #     results.append(
    #         test_utils.evaluate_model(
    #             model_name=name,
    #             model=model,
    #             X_test=X_test,
    #             y_true=y_true,
    #             logger=script_logger,
    #         )
    #     )



    # report_df = test_utils.classification_report_table(results)
    # script_logger.log_result(f"\n{report_df.round(3)}")

    # # -----------------------------------------------------------------------------
    # # Precision-Recall Curve
    # # -----------------------------------------------------------------------------

    # # precision, recall, thresholds = test_utils.prec_recall_curve(
    # #     y_true=y_true, probs=probabilities, logger=script_logger
    # # )

    # test_utils.plot_pr_grid(results=results)

    # # # -----------------------------------------------------------------------------
    # # # Optimal Threshold for MCC
    # # # -----------------------------------------------------------------------------
    # # script_logger.log_check("Finding optimal threshold for MCC...")
    # # best_mcc_threshold, best_mcc = test_utils.find_optimal_threshold_MCC(
    # #     y_true=y_true, probs=probabilities, logger=script_logger
    # # )

    # # 4. Generate the final report
    # # final_predictions = (probs >= best_threshold).astype(int)
    # # print(classification_report(y_true, final_predictions))
    # # test_utils.evaluate(
    # #     y_true=y_true,
    # #     predictions=predictions,
    # #     probabilities=probabilities,
    # #     threshold=best_mcc_threshold,
    # #     logger=script_logger,
    # # )

    # # -----------------------------------------------------------------------------
    # # ROC Curve
    # # -----------------------------------------------------------------------------

    # # test_utils.display_ROC_curve(
    # #     y_true=y_true, probabilities=probabilities, logger=script_logger
    # # )

    # test_utils.plot_roc_grid(results=results)
