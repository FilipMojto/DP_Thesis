

from pathlib import Path

import pandas as pd

from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER


def load_df(df_file_path: Path, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Loading the dataset...", print_to_console=True)

    df = pd.read_feather(df_file_path)
    logger.log_result(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns\n", print_to_console=True)

    return df