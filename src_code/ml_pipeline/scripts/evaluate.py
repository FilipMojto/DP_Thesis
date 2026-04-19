import argparse
from typing import Iterable, List, get_args

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skorch import NeuralNetClassifier
from xgboost import XGBClassifier
from notebooks.logging_config import MyLogger
from src_code.config import (
    ENGINEERED_DATA_DIR,
    EVALUATION_DIR,
    LOG_DIR,
    MODEL_DIR,
    SupportedModel,
)
from src_code.ml_pipeline.config import SUPPORTED_MODELS
import src_code.ml_pipeline.data_utils as dutls
from src_code.ml_pipeline.experimenting.types import EvalResults
from src_code.ml_pipeline.experimenting.utils import (
    get_experiment_dir,
    log_experiment_id,
)
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
    models: Iterable[SupportedModel] = SUPPORTED_MODELS,
    experiment_id: int = None,
):
    logger.start_session(
        session_id=experiment_id if experiment_id else MyLogger.DEF_SESSION_ID
    )
    log_experiment_id(logger=logger, experiment_id=experiment_id)
    exp_dir = (
        get_experiment_dir(experiment_id, target_dir=EVALUATION_DIR)
        if experiment_id
        else None
    )

    loaded_models = {
        model_type: dutls.load_artifact(
            dir=MODEL_DIR,
            artifact_type="trained_model",
            logger=logger,
            label=model_type,
        )
        for model_type in models
    }

    # results = []
    results_v2: List[EvalResults] = []
    # for risk-based assessment
    model_effort_scores = {}

    # =============================================================================
    # FINAL EVALUATION
    # =============================================================================

    test_df_versioner = VersionedFileManager(
        # file_path=PROCESSED_DATA_DIR / "test_transformed.feather", logger=logger
        file_path=ENGINEERED_DATA_DIR / "test_engineered.feather",
        logger=logger,
    )
    test_df = dutls.load_df(
        df_file_path=test_df_versioner.current_newest, logger=logger
    )

    test_df = drop_cols(df=test_df, cols=DROP_COLS, logger=logger)
    y_true = test_df["label"].values if "label" in test_df.columns else None
    # # -----------------------------------------------------------------------------
    # # Model Loading
    # # -----------------------------------------------------------------------------

    
    for name, artifact in loaded_models.items():
        logger.start_section(section_name=f"Evaluating model: {name}")
        model_wrapper = artifact.model_wrapper
     
        model_features = artifact.extract_features()
        logger.log_result(f"Model was trained on features: {len(model_features)}" if model_features is not None else "Model features not found in artifact.")

        if model_wrapper == None and model_features == None:
            raise ValueError(
                f"Model wrapper or features not found for model '{name}'. Cannot proceed with evaluation."
            )

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

        X_trans = model_wrapper.transform(test_df)

        # y_true = test_df["label"].values if "label" in test_df.columns else None
        y_proba = model_wrapper.model.predict_proba(X_trans)[:, 1]
        model_effort_scores[name] = y_proba

        # -----------------------------------------------------------------------------
        # Inference
        # -----------------------------------------------------------------------------

        y_true = test_df["label"] if "label" in test_df.columns else None
        results_local = test_utils.evaluate_model(
            model_name=name,
            model=model_wrapper.model,
            X_test=X_trans,
            y_true=y_true,
            probs=y_proba,
            logger=logger,
        )

        # results.append(results_local)
        results_v2.append(results_local)
    
    # -----------------------------------------------------------------------------
    # heuristic results for risk-based assessment
    # -----------------------------------------------------------------------------
    
    
    heuristics = {}

    if "loc_change" not in test_df.columns:
        if "loc_added" in test_df.columns and "loc_deleted" in test_df.columns:
            test_df["loc_change"] = test_df["loc_added"] + test_df["loc_deleted"]

    if "loc_change" in test_df.columns:
        heuristics["LOC_CHANGE"] = test_df["loc_change"].values

    if "files_changed" in test_df.columns:
        heuristics["FILES_CHANGED"] = test_df["files_changed"].values

    report_df = test_utils.classification_report_table(results_v2)
    logger.log_result(f"\n{report_df.round(3)}")

    if exp_dir:
        report_df.to_csv(exp_dir / "classification_report.csv")
        # Optional: Save a pretty version for humans
        with open(exp_dir / "report_summary.txt", "w") as f:
            f.write(report_df.to_string())
    

    # test_utils.plot_pr_grid(results=results, experiment_path=exp_dir)
    test_utils.plot_pr_combined(results=results_v2, experiment_path=exp_dir)
    # test_utils.plot_roc_grid(results=results, experiment_path=exp_dir)
    test_utils.plot_roc_combined(results=results_v2, experiment_path=exp_dir)
    test_utils.plot_effort_combined(
        model_scores=model_effort_scores,
        y_true=y_true,
        heuristic_scores=heuristics,
        experiment_path=exp_dir,
    )

    return results_v2


def get_parser(add_help: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluation script for ML & DL models.", add_help=add_help
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

    parser = get_parser(add_help=True)
    args = parser.parse_args()

    evaluate(logger=logger, models=args.models, experiment_id=None)
