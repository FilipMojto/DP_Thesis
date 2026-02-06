import argparse
from typing import get_args

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERING_MAPPINGS, EVALUATION_DIR, LOG_DIR, MODEL_DIR, PROCESSED_DATA_DIR, SupportedModels
from src_code.ml_pipeline.config import SUPPORTED_MODELS
import src_code.ml_pipeline.data_utils as dutls
from src_code.ml_pipeline.experimenting.utils import log_experiment_id
from src_code.ml_pipeline.preprocessing.feature_config import DROP_COLS
from src_code.ml_pipeline.preprocessing.preprocessing import drop_cols
import src_code.ml_pipeline.testing.testing as test_utils
from src_code.ml_pipeline.utils import get_experiment_dir
from src_code.utils.utils import timeit
from src_code.versioning import VersionedFileManager


DEF_SCRIPT_LOGGER = MyLogger(
    label="EVAL_DEPLOY",
    section_name="EVAL & DEPLOY LOGGER SCRIPT",
    file_log_path=LOG_DIR / "eval_deploy_log.log",
)

@timeit("Evaluation Phase", logger_param="logger")
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
    exp_dir = get_experiment_dir(experiment_id, target_dir=EVALUATION_DIR) if experiment_id else None
    
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


    test_df_versioner = VersionedFileManager(file_path=PROCESSED_DATA_DIR / "test_transformed.feather", logger=logger)
    test_df = dutls.load_df(df_file_path=test_df_versioner.current_newest, logger=logger)

    test_df = drop_cols(
        df=test_df, cols=DROP_COLS, logger=logger
    )
    # # -----------------------------------------------------------------------------
    # # Model Loading
    # # -----------------------------------------------------------------------------

   

    for name, verioner in models.items():
        # script_logger.log_check(f"Evaluating model: {name}")
        logger.start_section(section_name=f"Evaluating model: {name}")
        model = dutls.load_model(verioner.current_newest, logger)
        model_features = model.feature_names_in_

        if isinstance(model, RandomForestClassifier):
            logger.log_result("Loaded model is a Random Forest.")
        elif isinstance(model, XGBClassifier):
            logger.log_result("Loaded model is an XGBoost Classifier.")
        else:
            logger.log_result("Loaded model is of an unknown type.")

        # -----------------------------------------------------------------------------
        # Column Filtering
        # -----------------------------------------------------------------------------

        X_test = test_df[model_features]

        # -----------------------------------------------------------------------------
        # Inference
        # -----------------------------------------------------------------------------

        y_true = test_df["label"] if "label" in test_df.columns else None

        predictions, probabilities = test_utils.infer(
            X_test=X_test, model=model, logger=logger
        )

        results.append(
            test_utils.evaluate_model(
                model_name=name,
                model=model,
                X_test=X_test,
                y_true=y_true,
                logger=logger,
            )
        )



    report_df = test_utils.classification_report_table(results)
    logger.log_result(f"\n{report_df.round(3)}")

    if exp_dir:
        report_df.to_csv(exp_dir / "classification_report.csv")
        # Optional: Save a pretty version for humans
        with open(exp_dir / "report_summary.txt", "w") as f:
            f.write(report_df.to_string())


    test_utils.plot_pr_grid(results=results, experiment_path=exp_dir)
    test_utils.plot_roc_grid(results=results, experiment_path=exp_dir)



if __name__ == "__main__":
    logger = DEF_SCRIPT_LOGGER
    logger.start_session()

    evaluate(
        logger=logger
    )

    