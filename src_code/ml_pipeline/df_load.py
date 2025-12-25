

from pathlib import Path

import pandas as pd

from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER


def load_df(df_file_path: Path, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Loading the dataset...", print_to_console=True)

    df = pd.read_feather(df_file_path)
    logger.log_result(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns\n", print_to_console=True)

    return df


def save_df(df: pd.DataFrame, df_file_path: Path,logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Saving the preprocessed dataset...", print_to_console=True)

    # OUTPUT_PATH = PREPROCESSING_MAPPINGS[subset]['output']

    # 1. Get the names of the final features
    # feature_names = preprocessor.get_feature_names_out()

    # 2. Reconstruct the DataFrame
    # df_transformed = pd.DataFrame(df, columns=feature_names)

    df.to_feather(df_file_path)

    logger.log_result(f"Preprocessed dataset saved to {df_file_path}", print_to_console=True)