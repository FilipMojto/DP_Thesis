from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, get_args

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from notebooks.logging_config import MyLogger
from src_code.config import (
    ENGINEERED_DATA_DIR,
    EXTENDED_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    SubsetType,
)
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER

# from src_code.ml_pipeline.models import ModelWrapperBase
from src_code.ml_pipeline.models import ModelWrapperBase
from src_code.ml_pipeline.preprocessing.config import PreprocessMode
from src_code.versioning import VersionedFileManager


# class MyDataframe:
#     def __init__(self, dataframe: pd.DataFrame, src_path: Path):
#         self.dataframe = dataframe
#         self.src_path = src_path.absolute()
    
#     def data(self):
#         return self.dataframe
# class MyDataframe:
#     def __init__(self, dataframe: pd.DataFrame, src_path: Path):
#         self.df = dataframe
#         self.src_path = src_path.absolute()

#     def __getattr__(self, name):
#         # Redirect all unknown calls (like .shape, .head(), .iloc) to the pandas df
#         return getattr(self.df, name)

#     def __repr__(self):
#         return f"MyDataframe(src='{self.src_path}', rows={len(self.df)})"
@pd.api.extensions.register_dataframe_accessor("src")
class SourceAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._path = None

    def set_path(self, path: Path):
        self._path = path.absolute()
    
    def get_path(self):
        return self._path
    


def load_df(df_file_path: Path, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
    logger.log_check(
        f"Loading the dataset from {df_file_path.absolute()}...", print_to_console=True
    )

    df = pd.read_feather(df_file_path)
    logger.log_result(
        f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns\n",
        print_to_console=True,
    )

    # df.set_path(df_file_path)
    return df


def load_df_newest(df_file_path: Path, logger: MyLogger):
    input_df_versioner = VersionedFileManager(file_path=df_file_path, logger=logger)
    return load_df(df_file_path=input_df_versioner.current_newest, logger=logger)


# def load_input_dfs(
#     mode: PreprocessMode,
#     logger: MyLogger,
#     df_labels: Iterable[str] = get_args(SubsetType),
# ):
#     dfs: Dict[str, pd.DataFrame] = {}

#     for df_label in df_labels:
#         # target_dir = EXTENDED_DATA_DIR if mode == 'engineer' else ENGINEERED_DATA_DIR
#         file_path = (
#             EXTENDED_DATA_DIR / f"{df_label}_extended.feather"
#             if mode == "engineer"
#             else ENGINEERED_DATA_DIR / f"{df_label}_engineered.feather"
#         )

#         input_df_versioner = VersionedFileManager(file_path=file_path, logger=logger)
#         dfs[df_label] = load_df(
#             df_file_path=input_df_versioner.current_newest, logger=logger
#         )

#     return dfs


EDAMode = Literal["etl", "preprocessed"]


def load_input_dfs_eda(
    mode: EDAMode, logger: MyLogger, df_labels: Iterable[str] = get_args(SubsetType)
):
    dfs: Dict[SubsetType, pd.DataFrame] = {}

    for df_label in df_labels:
        # target_dir = EXTENDED_DATA_DIR if mode == 'engineer' else ENGINEERED_DATA_DIR
        file_path = (
            EXTENDED_DATA_DIR / f"{df_label}_extended.feather"
            if mode == "etl"
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

    pickled_instance = joblib.load(path)

    if isinstance(pickled_instance, ModelWrapperBase):
        model = pickled_instance.model
    elif isinstance(pickled_instance, Pipeline):
        model = pickled_instance
    elif isinstance(pickled_instance, Dict):
        model = pickled_instance
    else:
        raise ValueError("Unkown type of pickled instance!")
    # print(model.named_steps)

    # rf = model.named_steps['rf']
    # print("Pipeline Steps:", model.named_steps.keys())
    # logger.log_result(f"Hyperparameters: {model.get_params()}")
    # model_features = model.feature_names_in_
    # selector = model.named_steps['select_k']
    # selected_mask = selector.get_support()

    # selected_features = model.feature_names_in_[selected_mask]
    # print(selected_features)
    # features = None

    # if hasattr(model, "feature_names_in_"):
    #     features = model.feature_names_in_
    # elif hasattr(model, "named_steps"):
    #     for step in model.named_steps.values():
    #         if hasattr(step, "feature_names_in_"):
    #             features = step.feature_names_in_
    #             break
    #         if hasattr(step, "feature_names_"):
    #             features = step.feature_names_
    #             break
    # else:
    #     raise ValueError("Error")
    # if hasattr(model, "feature_names_"):
    #     features = model.feature_names_
    # elif hasattr(model, "named_steps"):
    #     final_estimator = list(model.named_steps.values())[-1]
    #     features = getattr(final_estimator, "feature_names_", None)
    # else:
    #     features = None

    # if features is not None:
    #     logger.log_result(
    #         f"Model trained on {len(features)} features:"
    #     )
    #     print(features)
    # else:
    #     logger.log_result("Feature names not found in loaded model.")
    # trained_features = model.feature_names_
    # logger.log_result(
    #     f"Model trained on {len(trained_features)} features:"
    # )

    features = None

    if hasattr(model, "feature_names_in_"):
        features = model.feature_names_in_
    elif hasattr(model, "feature_names_"):
        features = model.feature_names_
    else:
        logger.log_result("Unkown model features!")

    if features is not None:
        logger.log_result(f"The model was trained on {len(features)} features:")

    # logger.log_result(f"The model was trained on {len(model_features)} features:")

    # logger.log_result("Loading successful.")
    # # logger.log_result(f"Hyperparameters: {model.get_params()}")
    # logger.log_result(
    #     f"The model was trained on {len(features)} features:"
    # )

    return pickled_instance, features


ArtifactType = Literal["trained_model", "tuning-hyperparams"]


@dataclass
class PipelineArtifact:
    artifact_type: ArtifactType  # "tuning" | "trained_model"
    label: str = None
    model_wrapper: Optional[ModelWrapperBase] = None
    hyperparams: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    src_path: Optional[Path] = None

    def strip_prefix(self, prefix: str) -> dict:
        prefix_with_sep = prefix + "__"
        return {
            (
                key[len(prefix_with_sep) :] if key.startswith(prefix_with_sep) else key
            ): value
            for key, value in self.hyperparams.items()
        }

    def extract_features(self):
        wrapper = self.model_wrapper
        features = None

        if wrapper == None:
            return None

        # if hasattr(wrapper, "feature_names_in_"):
        #     features = wrapper.feature_names_in_
        if hasattr(wrapper, "feature_names_"):
            features = wrapper.feature_names_

        return features


# type PipelineArtifact = Dict | ModelWrapperBase


def __build_file_name(artifact_type: ArtifactType, label: str = None):
    return f"{label + "_" if label else ""}{artifact_type}.joblib"


# def load_artifact(path: Path, logger: MyLogger) -> PipelineArtifact:
def load_artifact(
    dir: Path, artifact_type: ArtifactType, logger: MyLogger, label: str = None
) -> PipelineArtifact:
    logger.log_check("Loading stored artifact...")

    versioner = VersionedFileManager(
        file_path=dir / __build_file_name(artifact_type=artifact_type, label=label),
        logger=logger,
        throw_not_found_err=True,
    )
    artifact: PipelineArtifact = joblib.load(versioner.current_newest)

    if not isinstance(artifact, PipelineArtifact):
        raise TypeError(
            f"Invalid artifact type: {type(artifact)}. " "Expected PipelineArtifact."
        )

    if artifact.artifact_type != artifact_type:
        raise ValueError("Incosistent artifact type!")

    logger.log_result(f"Loaded artifact type: {artifact.artifact_type}")
    return artifact


def save_artifact(dir: Path, artifact: PipelineArtifact, logger: MyLogger):
    versioner = VersionedFileManager(
        file_path=dir
        / __build_file_name(artifact_type=artifact.artifact_type, label=artifact.label),
        logger=logger,
    )
    path = versioner.next_base_output
    artifact.src_path = path

    logger.log_check(f"Saving artifact {artifact.artifact_type} to: {path}")
    joblib.dump(artifact, path)

    return path
