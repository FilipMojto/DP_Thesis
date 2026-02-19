import argparse
from typing import Dict, List, get_args

import pandas as pd
from notebooks.constants import TARGET
from notebooks.logging_config import MyLogger
from src_code.config import (
    EDA_DIR,
    EXTENDED_DATA_DIR,
    LOG_DIR,
    PROCESSED_DATA_DIR,
    SubsetType,
)
from src_code.ml_pipeline.EDA.correlations import plot_corr_matrix
from src_code.ml_pipeline.EDA.outliers import plot_boxplots_with_outliers
from src_code.ml_pipeline.EDA.plots import (
    plot_2D_embedding_separability,
    plot_categorical_comparison,
    plot_embedding_norm_distribution,
    plot_line_discrepancy_distribution,
    plot_num_feature_distributions,
    plot_pairwise_relationship,
)
from src_code.ml_pipeline.EDA.utils import NumFeatureSets
from src_code.ml_pipeline.data_utils import load_df, load_input_dfs, load_input_dfs_eda
from src_code.ml_pipeline.experimenting.types import EdaResults, MyDataset
from src_code.ml_pipeline.experimenting.utils import get_experiment_dir, log_experiment_id
from src_code.ml_pipeline.utils import describe_dataframe
from src_code.versioning import VersionedFileManager


DEF_SCRIPT_LOGGER = MyLogger(
    label="EDA", section_name="EDA", file_log_path=LOG_DIR / "EDA.log"
)


def perform_EDA(
    logger: MyLogger = DEF_SCRIPT_LOGGER,
    subset: SubsetType = "train",
    load_ETL_processed: bool = False,
    experiment_id: int = None,
    max_rows: int = None,
    intersect_with_processed=False,
):
    # if logger == DEF_SCRIPT_LOGGER:
    #     # If default logger is used, start a new session
    #     logger.start_session()
    logger.start_session(session_id=experiment_id if experiment_id else MyLogger.DEF_SESSION_ID)


    log_experiment_id(logger=logger, experiment_id=experiment_id)
    # logger.log_check(f"etl_processed: {load_ETL_processed}")
    logger.log_result(f"Config: {subset=}, {load_ETL_processed=}, {experiment_id=}, {max_rows=}, {intersect_with_processed=}")

    # input_df_path = (
    #     EXTENDED_DATA_DIR / f"{subset}_extended.feather"
    #     if load_ETL_processed
    #     else PROCESSED_DATA_DIR / f"{subset}_transformed.feather"
    # )
    # input_df_versioner = VersionedFileManager(file_path=input_df_path, logger=logger)
    # input_df = load_df(df_file_path=input_df_versioner.current_newest, logger=logger)

    # df_labels: List[SubsetType] = ["train", "test", "val"]
    # dfs: Dict[str, pd.DataFrame] = {}

    # for df_label in df_labels:
    #     df_path = (
    #         # EXTENDED_DATA_DIR / f"{df_label}_extended.feather"
    #         PROCESSED_DATA_DIR / f"{df_label}_engineered.feather"
    #         if load_ETL_processed
    #         else PROCESSED_DATA_DIR / f"{df_label}_transformed.feather"
    #     )
    #     df_versioner = VersionedFileManager(file_path=df_path, logger=logger)
    #     dfs[df_label] = load_df(df_file_path=df_versioner.current_newest, logger=logger)
    results = EdaResults()

    dfs = load_input_dfs_eda(
        mode="etl" if load_ETL_processed else "preprocessed", logger=logger
    )

    for label, df in dfs.items():
        nrows, ncols = df.shape
        results.loaded_datasets.append(MyDataset(type=label, rows=nrows, cols=ncols))

    if load_ETL_processed and intersect_with_processed:
        logger.log_check("Dropping cols not present in the processed data...")
        # Load the reference data
        dfs_processed = load_input_dfs_eda(mode="preprocessed", logger=logger)

        # Get the reference columns from the corresponding processed subset
        # Assuming 'subset' is the key you want to match against
        reference_cols = dfs_processed[subset].columns

        for df_label, df in dfs.items():
            # Find columns in current df that are NOT in reference_cols
            cols_to_drop = [c for c in df.columns if c not in reference_cols]

            if cols_to_drop:
                dfs[df_label] = df.drop(columns=cols_to_drop)
                logger.log_result(f"Dropped cols: {cols_to_drop}")
                logger.log_result(
                    f"Dropped {len(cols_to_drop)} columns from {df_label}"
                )

    for label, df in dfs.items():
        nrows, ncols = df.shape
        results.EDA_ready_datasets.append(MyDataset(type=label, rows=nrows, cols=ncols))

    input_df = dfs[subset]

    # train_df_path = (
    #     EXTENDED_DATA_DIR / f"train_extended.feather"
    #     if load_ETL_processed
    #     else PROCESSED_DATA_DIR / f"train_transformed.feather"
    # )
    # train_df_versioner = VersionedFileManager(file_path=train_df_path, logger=logger)
    # train_df = load_df(df_file_path=train_df_versioner.current_newest, logger=logger)

    # test_df_path = (
    #     EXTENDED_DATA_DIR / f"test_extended.feather"
    #     if load_ETL_processed
    #     else PROCESSED_DATA_DIR / f"test_transformed.feather"
    # )
    # test_df_versioner = VersionedFileManager(file_path=test_df_path, logger=logger)
    # test_df = load_df(df_file_path=test_df_versioner.current_newest, logger=logger)

    # val_df_path = (
    #     EXTENDED_DATA_DIR / f"test_extended.feather"
    #     if load_ETL_processed
    #     else PROCESSED_DATA_DIR / f"test_transformed.feather"
    # )
    # val_df_versioner = VersionedFileManager(file_path=val_df_path, logger=logger)
    # val_df = load_df(df_file_path=val_df_versioner.current_newest, logger=logger)

    # experiment_dir = EDA_DIR if experiment_id else None
    exp_dir = (
        get_experiment_dir(experiment_id, target_dir=EDA_DIR) if experiment_id else None
    )

    if max_rows and max_rows < len(input_df):
        logger.log_check(f"Limiting EDA dataset to {max_rows} rows.")
        # df_row_count = len(input_df)
        input_df = input_df.head(max_rows)
    else:
        logger.log_result(f"Not skipping any rows in dataset.")

    describe_dataframe(df=input_df, logger=logger, name=f"EDA {subset} dataframe")

    feature_sets = NumFeatureSets.extract_features(df=input_df, logger=logger)

    # plot_num_feature_distributions(
    #     input_df,
    #     logger=logger,
    #     feature_ctgs=feature_sets,
    #     drop_cols=["raise"],
    #     col_type="structural",
    #     experiment_dir=exp_dir / "struct_features_distributions",
    #     rows_per_page=5,
    # )
    # plot_num_feature_distributions(
    #     input_df,
    #     logger=logger,
    #     feature_ctgs=feature_sets,
    #     col_type="engineered",
    #     experiment_dir=exp_dir / "struct_features_engineered_distributions",
    #     rows_per_page=5,
    # )

    # plot_num_feature_distributions(input_df, logger=logger, feature_ctgs=feature_ctgs, cols=feature_ctgs.embedding_cols, experiment_dir=exp_dir / "embedding_distributions")
    # plot_num_feature_distributions(input_df, logger=logger, feature_ctgs=feature_ctgs, cols=feature_ctgs.tfidf_vectorized_cols, experiment_dir=exp_dir / "tfidf_vectorized_distributions")

    # NOTE: DONT DELETE!!

    # if not load_ETL_processed:
    #     logger.log_check("Plotting embeddings and vectors...")
    #     # plot_embedding_distributions(df=input_df, feature_ctgs=feature_ctgs, logger=logger, experiment_dir=exp_dir / "embedding_distributions")
    #     plot_num_feature_distributions(
    #         input_df,
    #         logger=logger,
    #         feature_ctgs=feature_ctgs,
    #         col_type="embedding",
    #         experiment_dir=exp_dir / "embedding_distributions",
    #     )
    #     plot_num_feature_distributions(
    #         input_df,
    #         logger=logger,
    #         feature_ctgs=feature_ctgs,
    #         col_type="vectorized",
    #         experiment_dir=exp_dir / "tfidf_vectorized_distributions",
    #     )

    # else:
    #     logger.log_result("ETL-processed data does not have embeddings!")

    # plot_feature_distribution(df=input_df, feature=TARGET, logger=logger, rotation=0, exp_dir=exp_dir / "other_distributions")

    # plot_embedding_norm_distribution(df=input_df, logger=logger, experiment_dir=exp_dir / "embedding_distributions")

    # -----------------------------------------------------------------------------
    # Other distributions
    # -----------------------------------------------------------------------------

    # plot_categorical_comparison(
    #     dfs=dfs,
    #     feature=TARGET,
    #     logger=logger,
    #     experiment_dir=exp_dir / "other_distributions",
    #     rows_per_page=3,
    # )

    # plot_categorical_comparison(
    #     dfs=dfs,
    #     feature="repo",
    #     logger=logger,
    #     experiment_dir=exp_dir / "other_distributions",
    #     rows_per_page=3,
    # )

    # for df in dfs.values():
    #     df["ext"] = df["filepath"].str.split(".").str[-1]

    # plot_categorical_comparison(
    #     dfs=dfs, feature="ext", logger=logger, experiment_dir=exp_dir / "other_distributions", rows_per_page=3
    # )

    # plot_line_discrepancy_distribution(df=input_df, logger=logger, experiment_dir=exp_dir / "other_distributions")

    # -----------------------------------------------------------------------------
    # Correlations
    # -----------------------------------------------------------------------------

    # logger.log_check("Generating corr matrix heatmaps...")

    # feature_sets
    # corr_config = {
    #     "numeric": {"data": feature_sets.numeric_cols, "top_n": 5, "proceed": True},
    #     "engineered": {"data": feature_sets.engineered_cols, "top_n": 5, "proceed": not load_ETL_processed},
    #     "embeddings": {"data": feature_sets.embedding_cols, "top_n": 5, "proceed": not load_ETL_processed},
    #     "tfidf": {"data": feature_sets.tfidf_vectorized_cols, "top_n": 5, "proceed": not load_ETL_processed},
    # }

    # for label, config in corr_config.items():
    #     if not config['proceed']:
    #         logger.log_result(f"Skipping {label} feature set...")
    #         continue
    #     else:
    #         logger.log_check(f"Proceeding with {label} feature set...")

    #     corr_matrix = input_df[
    #         [x for x in config["data"]] + [TARGET]
    #     ].corr()

    #     plot_corr_matrix(
    #         corr_matrix=corr_matrix,
    #         label=label,
    #         logger=logger,
    #         target=TARGET,
    #         top_n=config['top_n'],
    #         min_abs_corr=0.15,
    #         experiment_dir=exp_dir / "correlations",
    #     )

    plot_pairwise_relationship(
        df=input_df,
        feature_ctgs=feature_sets,
        logger=logger,
        experiment_dir=exp_dir / "correlations",
        top_features=3
    )

    return results

    # -----------------------------------------------------------------------------
    # Outliers
    # -----------------------------------------------------------------------------

    # logger.log_check("Starting the outlier phase...")

    # plot_boxplots_with_outliers(
    #     df=input_df,
    #     feature_sets=feature_sets,
    #     logger=logger,
    #     experiment_dir=exp_dir / "outliers",
    #     drop_binaries=True,
    # )

    # -----------------------------------------------------------------------------
    # Target Separability
    # -----------------------------------------------------------------------------
    
    # plot_2D_embedding_separability(df=input_df, logger=logger, sample_size=2000, experiment_dir=exp_dir / 'separability')


def get_parser():
    parser = argparse.ArgumentParser(description="EDA Script for ML pipeline.", add_help=False)

    parser.add_argument(
        "--subset",
        choices=get_args(SubsetType),
        default="train",
        required=False,
        help="Specify which subset (train, test or validate) to run through the pipeline.",
    )
    parser.add_argument(
        "--load-etl-df",
        action="store_true",
        required=False,
        default=False,
        help="Loads and uses ETL-processed dataset istead of classic pipeline-processed one.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        required=False,
        default=None,
        help="Limit dataset to first n rows only for testing purposes.",
    )
    parser.add_argument(
        "--intersect-with-processed",
        action="store_true",
        required=False,
        default=False,
        help="If unprocessed data is used, drop any column not present in the processed data.",
    )
    
    return parser

if __name__ == "__main__":
    # argparser = argparse.ArgumentParser(description="EDA Script for ML pipeline.")

    # argparser.add_argument(
    #     "--subset",
    #     choices=get_args(SubsetType),
    #     default="train",
    #     required=False,
    #     help="Specify which subset (train, test or validate) to run through the pipeline.",
    # )

    # argparser.add_argument(
    #     "--load-etl-df",
    #     action="store_true",
    #     required=False,
    #     default=False,
    #     help="Loads and uses ETL-processed dataset istead of classic pipeline-processed one.",
    # )

    # argparser.add_argument(
    #     "--max-rows",
    #     type=int,
    #     required=False,
    #     default=None,
    #     help="Limit dataset to first n rows only for testing purposes.",
    # )

    # argparser.add_argument(
    #     "--intersect-with-processed",
    #     action="store_true",
    #     required=False,
    #     default=False,
    #     help="If unprocessed data is used drop any column not present in the processed data.",
    # )
    parser = get_parser()
    args = parser.parse_args()


    perform_EDA(
        logger=DEF_SCRIPT_LOGGER,
        subset=args.subset,
        load_ETL_processed=args.load_etl_df,
        experiment_id=3,
        max_rows=args.max_rows,
        intersect_with_processed=args.intersect_with_processed,
    )
