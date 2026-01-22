import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER


def load_df(df_file_path: Path, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check(f"Loading the dataset from {df_file_path.absolute()}...", print_to_console=True)

    df = pd.read_feather(df_file_path)
    logger.log_result(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns\n", print_to_console=True)

    return df


def save_df(df: pd.DataFrame, df_file_path: Path, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Saving the preprocessed dataset...", print_to_console=True)

    # OUTPUT_PATH = PREPROCESSING_MAPPINGS[subset]['output']

    # 1. Get the names of the final features
    # feature_names = preprocessor.get_feature_names_out()

    # 2. Reconstruct the DataFrame
    # df_transformed = pd.DataFrame(df, columns=feature_names)

    df.to_feather(df_file_path)

    logger.log_result(f"Preprocessed dataset saved to {df_file_path}", print_to_console=True)


def save_model(model: BaseEstimator, path: Path, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
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
    logger.log_result(f"Hyperparameters: {model.get_params()}")
    logger.log_result(f"The model was trained on {len(model.feature_names_in_)} features:")

    return model
