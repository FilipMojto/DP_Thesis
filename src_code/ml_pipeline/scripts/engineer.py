from argparse import ArgumentParser
from typing import Dict, get_args
from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERED_DATA_DIR, LOG_DIR
from src_code.ml_pipeline.data_utils import save_df
from src_code.ml_pipeline.experimenting.types import DfMetadata, PreprocessingResults, SubsetArg
from src_code.ml_pipeline.experimenting.utils import log_experiment_id
from src_code.ml_pipeline.preprocessing.preprocessing import drop_invalid_rows, engineer_cols
from src_code.ml_pipeline.preprocessing.utils import load_dataframes
from src_code.ml_pipeline.utils import describe_dataframe, limit_dataframe_rows
from src_code.utils.utils import timeit
from src_code.versioning import VersionedFileManager


DEF_SCRIPT_LOGGER = MyLogger(
    label="ENGINEER",
    section_name="ENGINEER LOGGER SCRIPT",
    file_log_path=LOG_DIR / "engineer_log.log",
)

def get_parser(add_help: bool = False) -> ArgumentParser:
    parser = ArgumentParser(description="Early Preprocessing (Engineering) Phase Parser", add_help=add_help)
    
    parser.add_argument(
        "--subset",
        choices=get_args(SubsetArg),
        default="train",
        required=False,
        help="Specify which subset (train, test or validate) to run through the pipeline.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        required=False,
        default=None,
        help="Limit dataset to first n rows only for testing purposes.",
    )

    return parser


@timeit("Early Preprocessing Phase", logger_param="logger")
def early_preprocess(
    subset: SubsetArg,
    max_rows: int = None,
    experiment_id: int = None,
    logger: MyLogger = DEF_SCRIPT_LOGGER,
):
    # =============================================================================
    # PREPROCESSING
    # =============================================================================
    logger.start_session(session_id=experiment_id if experiment_id else MyLogger.DEF_SESSION_ID)
    log_experiment_id(logger=logger, experiment_id=experiment_id)
    logger.log_check(f"Config: {subset=}, {max_rows=}, {experiment_id=}")
    results = PreprocessingResults()

    input_dfs = load_dataframes(subset_arg=subset, mode='engineer', logger=logger)
    output_dfs_paths: Dict[str, VersionedFileManager] = {}
    results.loaded_datasets = input_dfs
    # for df in input_dfs:
    #     # nrows, ncols = df.shape
    #     results.loaded_datasets.append(df)
    
    for df in input_dfs:
        output_dfs_paths[df.metadata.type] = VersionedFileManager(
            file_path=ENGINEERED_DATA_DIR / f"{df.metadata.type}_engineered.feather",
            logger=logger,
        )
    
    for df in input_dfs:
        describe_dataframe(
            df=df.data, logger=logger, name=f"{subset} initial dataframe"
        )

        if max_rows:
            df.data = limit_dataframe_rows(
                df=df.data, script_logger=logger, max_rows=max_rows
            )


        # -----------------------------------------------------------------------------
        # Dropping invalid rows
        # -----------------------------------------------------------------------------

        df.data = drop_invalid_rows(
            df=df.data,
            row_filters={"time_since_last_change": lambda s: s >= 0},
        )

        before_engineer_cols = set(df.data.columns)

        logger.log_check("Starting data engineering subphase...")
        df.data = engineer_cols(target_df=df.data, logger=logger)
        after_engineer_cols = set(df.data.columns)
        
        logger.log_result(
            f"Engineered features: {after_engineer_cols - before_engineer_cols}",
            print_to_console=True,
        )
        logger.log_result("Data engineering subphase finished.")
        logger.log_result(
            f"Final dataframe shape: {df.data.shape}", print_to_console=True
        )

        output_path = output_dfs_paths[df.metadata.type].next_base_output

        nrows, ncols = df.data.shape
        results.preprocessed_datasets.append(DfMetadata(type=df.metadata.type, rows=nrows, cols=ncols, src_path=output_path))
    
        save_df(df=df.data, df_file_path=output_path, logger=logger)
        logger.log_result(f"Column data types: {df.data.dtypes.to_dict()}")

    return results


if __name__ == "__main__":
    logger = DEF_SCRIPT_LOGGER
    parser = get_parser(add_help=True)

    args = parser.parse_args()
    early_preprocess(subset=args.subset, max_rows=args.max_rows, experiment_id=None, logger=logger)