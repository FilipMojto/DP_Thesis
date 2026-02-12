import os
from pathlib import Path
from typing import Dict, Iterable, Literal, get_args

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from notebooks.logging_config import MyLogger
from src_code.config import (
    ENGINEERED_DATA_DIR,
    EXTENDED_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    SubsetType,
)
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.preprocessing.config import PreprocessMode
from src_code.versioning import VersionedFileManager


def load_df(df_file_path: Path, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check(
        f"Loading the dataset from {df_file_path.absolute()}...", print_to_console=True
    )

    df = pd.read_feather(df_file_path)
    logger.log_result(
        f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns\n",
        print_to_console=True,
    )

    return df


def load_df_newest(df_file_path: Path, logger: MyLogger):
    input_df_versioner = VersionedFileManager(file_path=df_file_path, logger=logger)
    return load_df(df_file_path=input_df_versioner.current_newest, logger=logger)


def load_input_dfs(
    mode: PreprocessMode,
    logger: MyLogger,
    df_labels: Iterable[str] = get_args(SubsetType),
):
    dfs: Dict[str, pd.DataFrame] = {}

    for df_label in df_labels:
        # target_dir = EXTENDED_DATA_DIR if mode == 'engineer' else ENGINEERED_DATA_DIR
        file_path = (
            EXTENDED_DATA_DIR / f"{df_label}_extended.feather"
            if mode == "engineer"
            else ENGINEERED_DATA_DIR / f"{df_label}_engineered.feather"
        )

        input_df_versioner = VersionedFileManager(file_path=file_path, logger=logger)
        dfs[df_label] = load_df(
            df_file_path=input_df_versioner.current_newest, logger=logger
        )

    return dfs


EDAMode = Literal["etl", "preprocessed"]


def load_input_dfs_eda(
    mode: EDAMode, logger: MyLogger, df_labels: Iterable[str] = get_args(SubsetType)
):
    dfs: Dict[str, pd.DataFrame] = {}

    for df_label in df_labels:
        # target_dir = EXTENDED_DATA_DIR if mode == 'engineer' else ENGINEERED_DATA_DIR
        file_path = (
            EXTENDED_DATA_DIR / f"{df_label}_extended.feather"
            if mode == 'etl'
            else TRANSFORMED_DATA_DIR / f"{df_label}_transformed.feather"
        )

        input_df_versioner = VersionedFileManager(file_path=file_path, logger=logger)
        dfs[df_label] = load_df(
            df_file_path=input_df_versioner.current_newest, logger=logger
        )

    return dfs


def save_df(
    df: pd.DataFrame, df_file_path: Path, logger: MyLogger = DEF_NOTEBOOK_LOGGER
):
    logger.log_check("Saving the preprocessed dataset...", print_to_console=True)

    # OUTPUT_PATH = PREPROCESSING_MAPPINGS[subset]['output']

    # 1. Get the names of the final features
    # feature_names = preprocessor.get_feature_names_out()

    # 2. Reconstruct the DataFrame
    # df_transformed = pd.DataFrame(df, columns=feature_names)

    df.to_feather(df_file_path)

    logger.log_result(
        f"Preprocessed dataset saved to {df_file_path}", print_to_console=True
    )


def save_model(
    model: BaseEstimator, path: Path, logger: MyLogger = DEF_NOTEBOOK_LOGGER
):
    # Ensure the directory exists
    logger.log_check("Saving the trained model...")
    os.makedirs("models", exist_ok=True)
    # MODEL_SAVE_PATH = MODEL_DIR / "random_forest_pipeline.joblib"
    # Save the entire fitted pipeline
    joblib.dump(model, path)
    logger.log_result(f"Saved to {path}.")


def load_model(path: Path, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Loading a trained model...")

    model = joblib.load(path)
    # rf = model.named_steps['rf']
    # print("Pipeline Steps:", model.named_steps.keys())
    # logger.log_result(f"Hyperparameters: {model.get_params()}")
    # model_features = model.feature_names_in_
    # logger.log_result(f"The model was trained on {len(model_features)} features:")

    logger.log_result("Loading successful.")
    # logger.log_result(f"Hyperparameters: {model.get_params()}")
    logger.log_result(
        f"The model was trained on {len(model.feature_names_in_)} features:"
    )

    return model
