import argparse
from typing import get_args

from git import List

from notebooks.logging_config import MyLogger
from src_code.config import LOG_DIR, PROCESSED_DATA_DIR, SubsetType, SupportedModels
from src_code.ml_pipeline.preprocess import preprocess
from src_code.ml_pipeline.tune import tune_hyperparams
from src_code.versioning import VersionedFileManager
from src_code.ml_pipeline.preprocess import transform_df

PIPELINE_CONFIG = [
    "early-preprocessing",
    "hyperparameter-tuning",
    "preprocessing",
    "training",
    "evaluation",
]

SUBSETS_TO_PREPROCESS: List[SubsetType] = ["train", "val", "test"]
MODELS_TO_TUNE: List[str] = ["RF", "XGB"]

# PROCESS_ARGS: List[str] = [*get_args(SubsetType), "all"]
# TUNE_ARGS: List[str] = [*get_args(SupportedModels), "all"]


def main(
    logger: MyLogger, filtered_phases=PIPELINE_CONFIG, models_to_tune=MODELS_TO_TUNE
):
    session_id = logger.start_session()
    preprocessed_train_path = PROCESSED_DATA_DIR / f"train_engineered.feather"

    for phase in filtered_phases:
        logger.log_check(f"Executing phase: {phase}")

        match phase:
            case "early-preprocessing":

                for subset in SUBSETS_TO_PREPROCESS:
                    preprocessed_train_path = preprocess(
                        subset=subset,
                        # engineer=True,
                        # transform=False,
                        experiment_id=session_id,
                    )

                    if subset == "train":
                        preprocessed_train_path = preprocessed_train_path

            case "hyperparameter-tuning":
                # here we load the newest preprocessed train data
                preprocessed_train_versioner = VersionedFileManager(
                    file_path=preprocessed_train_path, logger=logger
                )

                for model_type in models_to_tune:
                    logger.log_check(f"Tuning model: {model_type}")
                    tuned_model_path = tune_hyperparams(
                        preprocessed_df_path=preprocessed_train_versioner.current_newest,
                        model_type=model_type,
                    )
                    logger.log_result(f"Tuned model path: {tuned_model_path}")

                # tune_hyperparams(versioned_file_manager=preprocessed_train_versioner.current_newest
            case "preprocessing":
                # Preprocessing code here
                for subset in SUBSETS_TO_PREPROCESS:
                    transform_df(target_df=)
            case "training":
                # Training code here
                pass
            case "evaluation":
                # Evaluation code here
                pass

        # Here you would call the actual functions/modules for each phase
        # For example:
        # if phase == "early-preprocessing":
        #     early_preprocessing()
        # elif phase == "hyperparameter-tuning":
        #     hyperparameter_tuning()
        # ... and so on


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline Main Script")
    script_logger = MyLogger(
        label="ML_PIPELINE",
        section_name="ML PIPELINE LOGGER",
        file_log_path=LOG_DIR / "ml_pipeline_log.log",
    )

    parser.add_argument(
        "--phases",
        type=str,
        nargs="+",
        choices=PIPELINE_CONFIG,
        default=PIPELINE_CONFIG,
        help="Phases of the pipeline to execute.",
    )

    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        choices=MODELS_TO_TUNE,
        default=MODELS_TO_TUNE,
        help="Models to tune during the hyperparameter tuning phase.",
    )

    args = parser.parse_args()

    main(filtered_phases=args.phases, logger=script_logger)
