from typing import Dict, Iterable, List, get_args

import pandas as pd

from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERED_DATA_DIR, EXTENDED_DATA_DIR, SubsetType
from src_code.ml_pipeline.data_utils import load_df
from src_code.ml_pipeline.experimenting.types import DfMetadata, MyDataset, SubsetArg
from src_code.ml_pipeline.preprocessing.config import PreprocessMode
from src_code.versioning import VersionedFileManager


def load_input_df(
    mode: PreprocessMode,
    logger: MyLogger,
    df_label: str,
) -> MyDataset:
    # target_dir = EXTENDED_DATA_DIR if mode == 'engineer' else ENGINEERED_DATA_DIR
    file_path = (
        EXTENDED_DATA_DIR / f"{df_label}_extended.feather"
        if mode == "engineer"
        else ENGINEERED_DATA_DIR / f"{df_label}_engineered.feather"
    )

    input_df_versioner = VersionedFileManager(file_path=file_path, logger=logger)
    df = load_df(df_file_path=input_df_versioner.current_newest, logger=logger)
    # return DfMetadata(
    #     type=df_label,
    #     rows=df.shape[0],
    #     cols=df.shape[1],
    #     src_path=input_df_versioner.current_newest,
    # )
    return MyDataset(
        metadata=DfMetadata(
            type=df_label,
            rows=df.shape[0],
            cols=df.shape[1],
            src_path=input_df_versioner.current_newest,
        ),
        data=df,
    )



# def load_input_dfs(
#     mode: PreprocessMode,
#     logger: MyLogger,
#     df_labels: Iterable[str] = get_args(SubsetType),
# ):
#     dfs: Dict[str, pd.DataFrame] = {}

#     # for df_label in df_labels:
#     #     # target_dir = EXTENDED_DATA_DIR if mode == 'engineer' else ENGINEERED_DATA_DIR
#     #     file_path = (
#     #         EXTENDED_DATA_DIR / f"{df_label}_extended.feather"
#     #         if mode == "engineer"
#     #         else ENGINEERED_DATA_DIR / f"{df_label}_engineered.feather"
#     #     )

#     #     input_df_versioner = VersionedFileManager(file_path=file_path, logger=logger)
#     #     dfs[df_label] = load_df(
#     #         df_file_path=input_df_versioner.current_newest, logger=logger
#     #     )

#     for df_label in df_labels:
#         dfs[df_label] = load_input_df(mode=mode, logger=logger, df_label=df_label)

#     return dfs


def load_dataframes(
    subset_arg: SubsetArg,
    mode: PreprocessMode,
    logger: MyLogger,
    df_labels: Iterable[str] = get_args(SubsetType),
):
    # dfs: Dict[str, pd.DataFrame] = {}
    dfs: List[MyDataset] = []

    if subset_arg == "all":
        # dfs = dutls.load_input_dfs(mode=mode, logger=logger)
        for df_label in df_labels:
            dfs.append(load_input_df(mode=mode, logger=logger, df_label=df_label))
    else:

        # target_dir = ENGINEERED_DATA_DIR if mode == 'engineer' else TRANSFORMED_DATA_DIR
        # target_dir = EXTENDED_DATA_DIR if mode == 'engineer' else ENGINEERED_DATA_DIR
        # versioner = VersionedFileManager(file_path=target_dir / f"{subset_arg}_{mode}ed.", logger=logger)
        # dfs[subset_arg] = dutls.load_df(df_file_path=versioner.current_newest, logger=logger)
        # dfs[subset_arg] = load_input_df(mode=mode, logger=logger, df_label=subset_arg)
        dfs.append(load_input_df(mode=mode, logger=logger, df_label=subset_arg))

    return dfs
