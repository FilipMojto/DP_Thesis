import argparse
import time
from typing import Dict
from typing_extensions import get_args
from notebooks.constants import TARGET
from notebooks.logging_config import MyLogger
from src_code.config import (
    LOG_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    SubsetType,
)

from src_code.ml_pipeline.data_utils import save_df
import src_code.ml_pipeline.preprocessing.utils as dutls
from src_code.ml_pipeline.experimenting.types import SubsetArg
from src_code.ml_pipeline.experimenting.utils import log_experiment_id
import src_code.ml_pipeline.preprocessing.transform as tr
from src_code.ml_pipeline.scripts.train import RANDOM_STATE
from src_code.ml_pipeline.utils import describe_dataframe, limit_dataframe_rows
from src_code.utils.utils import timeit
from src_code.versioning import VersionedFileManager

DEF_SCRIPT_LOGGER = MyLogger(
    label="PREPROCESS",
    section_name="PREPROCESS LOGGER SCRIPT",
    file_log_path=LOG_DIR / "preprocess_log.log",
)


@timeit("Transformation Subphase", logger_param="script_logger")
def transform_df(
    subset: SubsetArg,
    max_rows: int = None,
    logger: MyLogger = DEF_SCRIPT_LOGGER,
    experiment_id: int = None,
):
    if logger == DEF_SCRIPT_LOGGER:
        # If default logger is used, start a new session
        logger.start_session()

    log_experiment_id(logger=logger, experiment_id=experiment_id)

    input_dfs = dutls.load_dataframes(
        subset_arg=subset, mode="transform", logger=logger
    )
    output_paths: Dict[str, VersionedFileManager] = {}

    # for df_label in input_dfs.keys():
    #     output_paths[df_label] = VersionedFileManager(
    #         file_path=PROCESSED_DATA_DIR / f"{df_label}_transformed.feather",
    #         logger=logger,
    #     )
    for df in input_dfs:
        output_paths[df.metadata.type] = VersionedFileManager(
            file_path=TRANSFORMED_DATA_DIR / f"{df.metadata.type}_transformed.feather",
            logger=logger,
        )

    # for df_label, target_df in input_dfs.items():
    for df in input_dfs:
        target_df = df.data
        feature_columns = [c for c in target_df.columns if c != TARGET]
        # feature_columns = target_df.columns.tolist()  # Use all columns for transformation, including target if present. The transformer should handle it appropriately.

        describe_dataframe(
            df=target_df, logger=logger, name=f"{subset} before transformation"
        )

        target_df = limit_dataframe_rows(
            df=target_df, script_logger=logger, max_rows=max_rows
        )
        target_df, fitted_transformer = tr.transform(
            df=target_df,
            subset=df.metadata.type,
            random_state=RANDOM_STATE,
            pandas_output=True,
            available_cols=feature_columns,
            target_col=TARGET
        )

        # --- Variance Explanation by Embeddings - Demo ---

        logger.log_result(
            f"Code embeddings explain "
            f"{tr.pca_explained_variance(fitted_transformer, 'code_embed'):.2%} of variance"
        )

        logger.log_result(
            f"Message embeddings explain "
            f"{tr.pca_explained_variance(fitted_transformer, 'msg_embed'):.2%} of variance"
        )

        logger.log_result("Dimensions of transformed dataframe: " + str(target_df.shape))
        logger.log_result(f"Column data types: {target_df.dtypes.to_dict()}")
        
        if TARGET not in target_df.columns:
            logger.log_result(f"Warning: Target column '{TARGET}' not found in transformed dataframe columns.", print_to_console=True)
        
        save_df(
            df=target_df,
            df_file_path=output_paths[df.metadata.type].next_base_output,
            logger=logger,
        )

    return [df_path.next_base_output for df_path in output_paths.values()]


def get_parser(add_help=False):
    parser = argparse.ArgumentParser(
        add_help=add_help, description="Preprocessing Script for ML Pipeline"
    )

    parser.add_argument(
        "--subset",
        choices=get_args(SubsetArg),
        default="train",
        required=False,
        help="Specify which subset (train, test or validate) to run through the pipeline.",
    )

    # parser.add_argument(
    #     "--engineer",
    #     action="store_true",
    #     help="Whether to perform data engineering after preprocessing.",
    # )

    # parser.add_argument(
    #     "--transform",
    #     action="store_true",
    #     help="Whether to perform transformations after preprocessing.",
    # )

    parser.add_argument(
        "--max-rows",
        type=int,
        required=False,
        default=None,
        help="Limit dataset to first n rows only for testing purposes.",
    )

    return parser


if __name__ == "__main__":
    start = time.time()

    parser = get_parser(add_help=True)
    args = parser.parse_args()
    subset: SubsetType = args.subset

    script_logger = DEF_SCRIPT_LOGGER

    # if args.transform:
    transform_df(subset=subset, max_rows=args.max_rows, logger=script_logger)
