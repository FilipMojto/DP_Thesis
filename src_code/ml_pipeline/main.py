import argparse
import json
from pathlib import Path
from typing import get_args

from git import List

from notebooks.logging_config import MyLogger
from src_code.config import (
    LOG_DIR,
    ML_PIPELINE_DIR,
    PROCESSED_DATA_DIR,
    SubsetType,
    SupportedModels,
)
from src_code.ml_pipeline.scripts.evaluate import evaluate
from src_code.ml_pipeline.scripts.preprocess import early_preprocess
from src_code.ml_pipeline.scripts.train import train
from src_code.ml_pipeline.scripts.tune import tune_hyperparams
from src_code.versioning import VersionedFileManager
from src_code.ml_pipeline.scripts.preprocess import transform_df

PIPELINE_CONFIG = [
    "early-preprocessing",
    "hyperparameter-tuning",
    "preprocessing",
    "training",
    "evaluation",
]

SUBSETS_TO_PREPROCESS: List[SubsetType] = ["train", "val", "test"]
MODELS_TO_TUNE: List[str] = ["RF", "XGB"]
MODELS_TO_TRAIN: List[str] = ["RF", "XGB"]

# PROCESS_ARGS: List[str] = [*get_args(SubsetType), "all"]
# TUNE_ARGS: List[str] = [*get_args(SupportedModels), "all"]


def main(
    logger: MyLogger,
    # filtered_phases=PIPELINE_CONFIG,
    models_to_tune=MODELS_TO_TUNE,
    tune_kwargs=None,
    subsets_to_preprocess=SUBSETS_TO_PREPROCESS,
    engineering_kwargs=None,
    transformation_kwargs=None,
    models_to_train=MODELS_TO_TRAIN,
    training_kwargs=None,
    to_evaluate: bool = False,
    evaluate_kwargs=None,
):
    session_id = logger.start_session()
    preprocessed_train_path = PROCESSED_DATA_DIR / f"train_engineered.feather"
    logger.logger.info(f"Session ID: {session_id}")
    # logger.log_check(f"Filtered Phases to execute: {filtered_phases}")
    logger.logger.info(f"Models to Tune: {models_to_tune}")
    logger.logger.info(f"Subsets to Preprocess: {subsets_to_preprocess}")
    logger.logger.info(f"Models to Train: {models_to_train}")
    logger.logger.info(f"Evaluate: {to_evaluate}")
    # for phase in filtered_phases:
    #     logger.log_check(f"Executing phase: {phase}")

    # match phase:
    #     case "early-preprocessing":

    for subset in subsets_to_preprocess:
        preprocessed_train_path_local = early_preprocess(
            subset=subset,
            # engineer=True,
            # transform=False,
            experiment_id=session_id,
            **(engineering_kwargs or {}),
        )

        if subset == "train":
            preprocessed_train_path = preprocessed_train_path_local

    # case "hyperparameter-tuning":
    # here we load the newest preprocessed train data
    preprocessed_train_versioner = VersionedFileManager(
        file_path=preprocessed_train_path, logger=logger
    )

    for model_type in models_to_tune:
        logger.log_check(f"Tuning model: {model_type}")
        tuned_model_path = tune_hyperparams(
            preprocessed_df_path=preprocessed_train_versioner.current_newest,
            model_type=model_type,
            **(tune_kwargs or {}),
        )
        logger.log_result(f"Tuned model path: {tuned_model_path}")

    # tune_hyperparams(versioned_file_manager=preprocessed_train_versioner.current_newest
    # case "preprocessing":
    # Preprocessing code here
    for subset in subsets_to_preprocess:
        # df_file_versioner = VersionedFileManager(file_path=PROCESSED_DATA_DIR / f"{subset}_engineered.feather")
        transform_df(
            subset=subset, experiment_id=session_id, **(transformation_kwargs or {})
        )
    # case "training":

    for model in models_to_train:
        logger.log_check(f"Training model: {model}")
        trained_model_path = train(
            model_type=model,
            # script_logger=logger,
            # load_tuned=True,
            experiment_id=session_id,
            **(training_kwargs or {}),
        )
        logger.log_result(f"Trained model path: {trained_model_path}")
    # case "evaluation":
    # Evaluation code here
    if to_evaluate:
        evaluate(experiment_id=session_id, **(evaluate_kwargs or {}))

        # Here you would call the actual functions/modules for each phase
        # For example:
        # if phase == "early-preprocessing":
        #     early_preprocessing()
        # elif phase == "hyperparameter-tuning":
        #     hyperparameter_tuning()
        # ... and so on


def process_pipeline_config(json_path: Path, script_logger: MyLogger) -> dict:
    with open(json_path, "r") as f:
        config_data = json.load(f)

        # early_preprocessing_subsets = []
        hyperparameter_tuning_models = []
        preprocessing_subsets = []
        training_models = []
        # evaluation_metrics = []
        evaluate = False

        tune_hyperparams_kwargs = {}
        engineering_kwargs = {}
        transformation_kwargs = {}
        training_kwargs = {}
        evaluate_kwargs = {}

        for phase in config_data.get("phases", []):
            phase_name = phase.get("name")

            if phase.get("skip", False):
                script_logger.log_result(f"Skipping phase: {phase_name}")
                continue

            match phase_name:
                # case "early-preprocessing":
                #     early_preprocessing_subsets = phase.get("subsets", [])
                case "hyperparameter-tuning":
                    hyperparameter_tuning_models = phase.get("models", [])
                    tune_hyperparams_kwargs = phase.get("kwargs", {})
                case "preprocessing":
                    preprocessing_subsets = phase.get("subsets", [])
                    engineering_kwargs = phase.get("engineering_kwargs", {})
                    transformation_kwargs = phase.get("transformation_kwargs", {})

                case "training":
                    training_models = phase.get("models", [])
                    training_kwargs = phase.get("kwargs", {})
                case "evaluation":
                    evaluate = True
                    evaluate_kwargs = phase.get("kwargs", {})
                case _:
                    raise ValueError(f"Unknown phase name: {phase_name}")

    return {
        # "early_preprocessing_subsets": early_preprocessing_subsets,
        "hyperparameter_tuning_models": (
            hyperparameter_tuning_models,
            tune_hyperparams_kwargs,
        ),
        "preprocessing_subsets": (
            preprocessing_subsets,
            engineering_kwargs,
            transformation_kwargs,
        ),
        "training_models": (training_models, training_kwargs),
        "evaluate": (evaluate, evaluate_kwargs),
        # "evaluation_metrics": evaluation_metrics,
    }


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="ML Pipeline Main Script")
    script_logger = MyLogger(
        label="ML_PIPELINE",
        section_name="ML PIPELINE LOGGER",
        file_log_path=LOG_DIR / "ml_pipeline_log.log",
    )

    json_dump = ML_PIPELINE_DIR / "pipeline_config.json"
    # with open(json_dump, "r") as f:
    #     config_data = json.load(f)

    # parser.add_argument(
    #     "--phases",
    #     type=str,
    #     nargs="+",
    #     choices=PIPELINE_CONFIG,
    #     default=PIPELINE_CONFIG,
    #     help="Phases of the pipeline to execute.",
    # )

    # parser.add_argument(
    #     "--model-tune",
    #     type=str,
    #     nargs="+",
    #     choices=MODELS_TO_TUNE,
    #     default=MODELS_TO_TUNE,
    #     help="Models to tune during the hyperparameter tuning phase.",
    # )

    # parser.add_argument(
    #     "--model-train",
    #     type=str,
    #     nargs="+",
    #     choices=MODELS_TO_TRAIN,
    #     default=MODELS_TO_TRAIN,
    #     help="Models to train during the training phase.",
    # )

    # args = parser.parse_args()
    config = process_pipeline_config(json_path=json_dump, script_logger=script_logger)

    main(
        logger=script_logger,
        # filtered_phases=args.phases,
        models_to_tune=config["hyperparameter_tuning_models"][0],
        tune_kwargs=config["hyperparameter_tuning_models"][1],
        subsets_to_preprocess=config["preprocessing_subsets"][0],
        engineering_kwargs=config["preprocessing_subsets"][1],
        transformation_kwargs=config["preprocessing_subsets"][2],
        models_to_train=config["training_models"][0],
        training_kwargs=config["training_models"][1],
        to_evaluate=config["evaluate"][0],
        evaluate_kwargs=config["evaluate"][1],
    )
