from argparse import ArgumentParser
from typing import Dict, List, get_args
from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERED_DATA_DIR, EXTENDED_DATA_DIR, LOG_DIR, PROCESSED_DATA_DIR, SubsetType
from src_code.ml_pipeline.data_utils import load_df, save_df
from src_code.ml_pipeline.experimenting.types import SubsetArg
from src_code.ml_pipeline.experimenting.utils import log_experiment_id
from src_code.ml_pipeline.preprocessing.preprocessing import drop_invalid_rows, engineer_cols
from src_code.ml_pipeline.scripts.preprocess import load_dataframes
from src_code.ml_pipeline.utils import describe_dataframe, limit_dataframe_rows
from src_code.utils.utils import timeit
from src_code.versioning import VersionedFileManager


DEF_SCRIPT_LOGGER = MyLogger(
    label="ENGINEER",
    section_name="ENGINEER LOGGER SCRIPT",
    file_log_path=LOG_DIR / "engineer_log.log",
)

def get_parser():
    parser = ArgumentParser(description="Early Preprocessing (Engineering) Phase Parser", add_help=False)
    
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


    # if logger == DEF_SCRIPT_LOGGER:
    #     # If default logger is used, start a new session
    #     logger.start_session()

    # script_logger.log_check("Starting preprocessing phase...")
    # script_logger.log_check(
    #     f"Experiment ID: {experiment_id}"
    #     if experiment_id
    #     else "No Experiment ID provided."
    # )
    log_experiment_id(logger=logger, experiment_id=experiment_id)
    logger.log_check(f"Subset: {subset}")
    # target_df_path = TARGET_DF_FILE = PREPROCESSING_MAPPINGS[subset]["input"]
    input_dfs = load_dataframes(subset_arg=subset, mode='engineer', logger=logger)
    output_dfs_paths: Dict[str, VersionedFileManager] = {}
    
    for df_label in input_dfs.keys():
        output_dfs_paths[df_label] = VersionedFileManager(
            file_path=ENGINEERED_DATA_DIR / f"{df_label}_engineered.feather",
            logger=logger,
        )
        # output_dfs_paths.append(VersionedFileManager(
        #     file_path=ENGINEERED_DATA_DIR / f"{subset}_engineered.feather",
        #     logger=script_logger,
        # ))
    # input_df_file = VersionedFileManager(
    #     file_path=EXTENDED_DATA_DIR / f"{subset}_extended.feather", logger=script_logger
    # )
    # output_df_file = VersionedFileManager(
    #     file_path=ENGINEERED_DATA_DIR / f"{subset}_engineered.feather",
    #     logger=script_logger,
    # )

    # target_df_path = TARGET_DF_FILE = PREPROCESSING_MAPPINGS[subset]["input"]
    for df_label, target_df in input_dfs.items():
        # target_df_path = input_df_file.current_newest
        # target_df = load_df(target_df_path)

        # script_logger.log_result(
        #     f"Initial dataframe shape: {target_df.shape}", print_to_console=True
        # )
        describe_dataframe(
            df=target_df, logger=logger, name=f"{subset} initial dataframe"
        )

        # if max_rows is not None:
        #     script_logger.log_check(f"Limiting to first {max_rows} rows for testing...")
        #     target_df = target_df.head(max_rows)
        #     script_logger.log_result(
        #         f"Dataframe shape after row limit: {target_df.shape}",
        #         print_to_console=True,
        #     )

        if max_rows:
            target_df = limit_dataframe_rows(
                df=target_df, script_logger=logger, max_rows=max_rows
            )

        # # -----------------------------------------------------------------------------
        # # Dropping invalid cols
        # # -----------------------------------------------------------------------------

        # target_df = prep.drop_cols(
        #     df=target_df, cols=ftr_cfg.DROP_COLS, logger=script_logger
        # )

        # -----------------------------------------------------------------------------
        # Dropping invalid rows
        # -----------------------------------------------------------------------------

        target_df = drop_invalid_rows(
            df=target_df,
            # numeric_features=NUMERIC_FEATURES,
            # row_filters={"time_since_last_change": target_df["time_since_last_change"] < 0},
            row_filters={"time_since_last_change": lambda s: s >= 0},
        )

        # -----------------------------------------------------------------------------
        # Data Engineering
        # -----------------------------------------------------------------------------

        # script_logger.log_check("Starting data engineering subphase...")
        # # -----------------------------------------------------------------------------
        # # Feature Derivation
        # # -----------------------------------------------------------------------------

        # # mappings = {
        # #     "loc_churn_ratio": lambda df: df["loc_added"] / (df["loc_deleted"] + 1),
        # #     "activity_per_exp": lambda df: df["author_recent_activity_pre"]
        # #     / (df["author_exp_pre"] + 1),
        # # }

        # # [STAGE 1] Derived Features
        # target_df = de.create_derived_features(
        #     df=target_df, mappings=ftr_cfg.DERIVED_FEATURES
        # )
        # # [STAGE 2] Creating Buckets
        # target_df = de.create_buckets(
        #     df=target_df, mappings=ftr_cfg.BUCKET_MAPPINGS, encode=True
        # )
        # # [STAGE 3] Aggregating line token features
        # target_df = de.aggr_line_token_features(df=target_df, features=LINE_TOKEN_FEATURES)
        # # [STAGE 4] Feature interactions
        # target_df = de.create_feature_interactions(
        #     df=target_df, features=INTERACTION_FEATURES
        # )

        # script_logger.log_result("Data engineering subphase finished.")
        # if engineer:
        before_engineer_cols = set(target_df.columns)

        logger.log_check("Starting data engineering subphase...")
        target_df = engineer_cols(target_df=target_df, logger=logger)
        # SCRIPT_LOGGER.log_result(f"Engineered features: {ENGINEERED_FEATURES}", print_to_console=True)
        after_engineer_cols = set(target_df.columns)
        logger.log_result(
            f"Engineered features: {after_engineer_cols - before_engineer_cols}",
            print_to_console=True,
        )
        logger.log_result("Data engineering subphase finished.")

        # -----------------------------------------------------------------------------
        # # Tuning
        # # -----------------------------------------------------------------------------

        # tune_main()

        # -----------------------------------------------------------------------------
        # Transformations
        # -----------------------------------------------------------------------------

        # if transform:
        # script_logger.log_check("Starting transformations subphase...")
        # target_df, fitted_transformer = tr.transform(
        #     df=target_df,
        #     subset=subset,
        #     random_state=RANDOM_STATE,
        # )

        # # --- Variance Explanation by Embeddings - Demo ---

        # script_logger.log_result(
        #     f"Code embeddings explain "
        #     f"{tr.pca_explained_variance(fitted_transformer, 'code_embed'):.2%} of variance"
        # )

        # script_logger.log_result(
        #     f"Message embeddings explain "
        #     f"{tr.pca_explained_variance(fitted_transformer, 'msg_embed'):.2%} of variance"
        # )
        # script_logger.log_result("Transformations subphase finished.")

        # script_logger.log_result("Preprocessing phase finished.")
        # end = time.time()
        # script_logger.log_result(f"Preprocessing time: {end - start:.2f} seconds.")
        logger.log_result(
            f"Final dataframe shape: {target_df.shape}", print_to_console=True
        )

        # dutls.save_df(df=target_df, df_file_path=ENGINEERING_MAPPINGS[subset]["output"])
        # save_df(df=target_df, df_file_path=output_df_file.next_base_output)
        save_df(df=target_df, df_file_path=output_dfs_paths[df_label].next_base_output)


        logger.log_result(f"Column data types: {target_df.dtypes.to_dict()}")
    
    return [output_df_file.next_base_output for output_df_file in output_dfs_paths.values()]


if __name__ == "__main__":
    logger = DEF_SCRIPT_LOGGER
    parser = get_parser()

    args = parser.parse_args()
    early_preprocess(subset=args.subset, max_rows=args.max_rows, experiment_id=None, logger=logger)