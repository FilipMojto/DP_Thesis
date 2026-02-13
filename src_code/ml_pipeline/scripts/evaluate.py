import argparse
from typing import get_args

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skorch import NeuralNetClassifier
from xgboost import XGBClassifier
from notebooks.logging_config import MyLogger
from src_code.config import (
    ENGINEERED_DATA_DIR,
    ENGINEERING_MAPPINGS,
    EVALUATION_DIR,
    LOG_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    SupportedModel,
)
from src_code.ml_pipeline.config import SUPPORTED_MODELS
import src_code.ml_pipeline.data_utils as dutls
from src_code.ml_pipeline.experimenting.utils import (
    get_experiment_dir,
    log_experiment_id,
)
from src_code.ml_pipeline.models import ModelWrapperBase
from src_code.ml_pipeline.preprocessing.feature_config import DROP_COLS
from src_code.ml_pipeline.preprocessing.preprocessing import drop_cols
import src_code.ml_pipeline.testing.testing as test_utils
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
    models: list = SUPPORTED_MODELS,
    experiment_id: int = None,
):
    # MODELS = {
    #     "Random Forest": VersionedFileManager(file_path=MODEL_DIR / "RF_model_train.joblib", logger=logger),
    #     "XGBoost": VersionedFileManager(file_path=MODEL_DIR / "XGB_model_train.joblib", logger=logger),
    # }
    logger.start_session(session_id=experiment_id if experiment_id else MyLogger.DEF_SESSION_ID)
    log_experiment_id(logger=logger, experiment_id=experiment_id)
    exp_dir = (
        get_experiment_dir(experiment_id, target_dir=EVALUATION_DIR)
        if experiment_id
        else None
    )

    loaded_models = {
        model_type: VersionedFileManager(
            file_path=MODEL_DIR / f"{model_type}_model_train.joblib",
            logger=logger,
        )
        for model_type in models
    }

    results = []

    # =============================================================================
    # FINAL EVALUATION
    # =============================================================================

    test_df_versioner = VersionedFileManager(
        # file_path=PROCESSED_DATA_DIR / "test_transformed.feather", logger=logger
        file_path=ENGINEERED_DATA_DIR / "test_engineered.feather" , logger=logger
    )
    test_df = dutls.load_df(
        df_file_path=test_df_versioner.current_newest, logger=logger
    )

    test_df = drop_cols(df=test_df, cols=DROP_COLS, logger=logger)
    # # -----------------------------------------------------------------------------
    # # Model Loading
    # # -----------------------------------------------------------------------------

    for name, versioner in loaded_models.items():
        # script_logger.log_check(f"Evaluating model: {name}")
        logger.start_section(section_name=f"Evaluating model: {name}")
        model_wrapper, model_features = dutls.load_model(versioner.current_newest, logger)
        # model_features = model.feature_names_
        
        if isinstance(model_wrapper, RandomForestClassifier):
            logger.log_result("Loaded model is a Random Forest.")
        elif isinstance(model_wrapper, XGBClassifier):
            logger.log_result("Loaded model is an XGBoost Classifier.")
        elif isinstance(model_wrapper, NeuralNetClassifier):
            logger.log_result("Loaded model is an Neural Net Classifier.")
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

        # predictions, probabilities = test_utils.infer(
        #     X_test=X_test, model=model_wrapper.model, logger=logger
        # )

        X_trans = model_wrapper.pipeline[:-1].transform(X_test).astype(np.float32)

        results.append(
            test_utils.evaluate_model(
                model_name=name,
                model=model_wrapper.model,
                X_test=X_trans,
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


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluation script for ML & DL models.", add_help=False
    )

    parser.add_argument(
        "--models",
        choices=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
        required=False,
        nargs="+",
        help="One or models to evaluate",
    )

    return parser


if __name__ == "__main__":
    logger = DEF_SCRIPT_LOGGER
    # logger.start_session()

    parser = get_parser()
    args = parser.parse_args()

    evaluate(logger=logger, models=args.models, experiment_id=None)
