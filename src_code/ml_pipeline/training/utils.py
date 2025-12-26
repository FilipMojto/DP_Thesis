<<<<<<< HEAD
=======


>>>>>>> e94287b6ec2327bd8a597cfb3be9e88a30cf7447
from typing import Iterable
import pandas as pd

from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER


<<<<<<< HEAD
def drop_cols(
    df: pd.DataFrame, cols: Iterable[str], logger: NotebookLogger = DEF_NOTEBOOK_LOGGER
):
    logger.log_check("Dropping the specified columns...")

=======
def drop_cols(df: pd.DataFrame, cols: Iterable[str], logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Dropping the specified columns...")
    
>>>>>>> e94287b6ec2327bd8a597cfb3be9e88a30cf7447
    start_cols = set(df.columns)

    df = df.drop(columns=cols, errors="ignore")

    end_cols = set(df.columns)

    logger.log_result("Dropping completed.")
    logger.log_result(f"Columns dropped: {len(start_cols - end_cols)}")
    logger.log_result(f"Columns remaining: {len(end_cols)}")
    return df


<<<<<<< HEAD
def analyze_features(
    df: pd.DataFrame, target: str, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER
):
    logger.log_check("Starting df feature analysis...")
    # numeric_features = df.select_dtypes(include=["float64", "int64", "int8"]).columns.tolist()

    # numeric_features = df.select_dtypes(include="number").columns.tolist()
    numeric_features = df.select_dtypes(include=["number", "bool"]).columns.tolist()

=======
def analyze_features(df: pd.DataFrame, target: str, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check("Starting df feature analysis...")
    numeric_features = df.select_dtypes(include=["float64", "int64", "int8"]).columns.tolist()
>>>>>>> e94287b6ec2327bd8a597cfb3be9e88a30cf7447
    numeric_features.remove(target)
    logger.log_result(f"Numeric features: {numeric_features}", print_to_console=True)

    categorical_features = df.select_dtypes(include=["category"]).columns.tolist()
<<<<<<< HEAD
    logger.log_result(
        f"Categorical features: {categorical_features}", print_to_console=True
    )

    structured_features = [
        f for f in numeric_features if not f.startswith(("code_emb_", "msg_emb_"))
    ]
    logger.log_result(
        f"Structural features: {structured_features}", print_to_console=True
    )
    logger.log_result(len(structured_features), print_to_console=True)

    embedding_features = [
        f for f in numeric_features if f.startswith(("code_emb_", "msg_emb_"))
    ]
    logger.log_result(
        f"embedding_features: {embedding_features}", print_to_console=True
    )
=======
    logger.log_result(f"Categorical features: {categorical_features}", print_to_console=True)

    structured_features = [
        f for f in numeric_features
        if not f.startswith(("code_emb_", "msg_emb_"))
    ]
    logger.log_result(f"Structural features: {structured_features}", print_to_console=True)
    logger.log_result(len(structured_features), print_to_console=True)


    embedding_features = [
        f for f in numeric_features
        if f.startswith(("code_emb_", "msg_emb_"))
    ]
    logger.log_result(f"embedding_features: {embedding_features}", print_to_console=True)
>>>>>>> e94287b6ec2327bd8a597cfb3be9e88a30cf7447

    logger.log_result("Analysis completed.")
