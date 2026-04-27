import argparse
from typing import Iterable, get_args

import pandas as pd

from notebooks.logging_config import MyLogger
from src_code.config import (
    EDA_DIR,
    LOG_DIR,
    SubsetType,
)
from src_code.ml_pipeline.EDA.outliers import plot_boxplots_with_outliers
from src_code.ml_pipeline.EDA.plots import (
    plot_2D_embedding_separability,
    plot_line_discrepancy_distribution,
    plot_num_feature_distributions,
    plot_pairwise_relationship,
)
from src_code.ml_pipeline.EDA.utils import NumFeatureSets
from src_code.ml_pipeline.data_utils import load_input_dfs_eda
from src_code.ml_pipeline.experimenting.types import EdaResults, DfMetadata
from src_code.ml_pipeline.experimenting.utils import (
    get_experiment_dir,
    log_experiment_id,
)
from src_code.ml_pipeline.utils import describe_dataframe


DEF_SCRIPT_LOGGER = MyLogger(
    label="EDA", section_name="EDA", file_log_path=LOG_DIR / "EDA.log"
)


def inspect_dataframe(df: pd.DataFrame, logger: MyLogger, name: str = "DataFrame"):
    logger.log_result(f"Inspecting {name}...")
    logger.log_result(f"Dataframe shape: {df.shape}", print_to_console=True)
    logger.log_result(f"Dataframe columns: {df.columns.tolist()}")
    logger.log_result(f"Dataframe dtypes:\n{df.dtypes}")


def identify_missing_values(df: pd.DataFrame, logger: MyLogger):
    logger.log_result("Identifying missing values in the dataframe...")
    missing_values = df.isnull().sum()
    total_rows = len(df)
    missing_report = missing_values[missing_values > 0].to_dict()

    if missing_report:
        logger.log_result(f"Missing values found in the following columns:")
        for col, count in missing_report.items():
            percent = (count / total_rows) * 100
            logger.log_result(f" - {col}: {count} missing ({percent:.2f}%)")
    else:
        logger.log_result("No missing values found in the dataframe.")


def perform_EDA(
    logger: MyLogger = DEF_SCRIPT_LOGGER,
    subset: SubsetType = "train",
    load_ETL_processed: bool = False,
    experiment_id: int = None,
    max_rows: int = None,
    intersect_with_processed=False,
    limit_features: Iterable[str] = None,
):
    logger.start_session(
        session_id=experiment_id if experiment_id else MyLogger.DEF_SESSION_ID
    )

    log_experiment_id(logger=logger, experiment_id=experiment_id)
    logger.log_result(
        f"Config: {subset=}, {load_ETL_processed=}, {experiment_id=}, {max_rows=}, {intersect_with_processed=}"
    )
    results = EdaResults()

    mode = "etl" if load_ETL_processed else "preprocessed"

    dfs = load_input_dfs_eda(mode=mode, logger=logger)

    for label, df in dfs.items():
        nrows, ncols = df.shape
        results.loaded_datasets.append(DfMetadata(type=label, rows=nrows, cols=ncols))

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
        results.EDA_ready_datasets.append(
            DfMetadata(type=label, rows=nrows, cols=ncols)
        )

    input_df = dfs[subset]

    # experiment_dir = EDA_DIR if experiment_id else None
    exp_dir = (
        get_experiment_dir(experiment_id, target_dir=EDA_DIR, label=mode)
        if experiment_id
        else None
    )

    if max_rows and max_rows < len(input_df):
        logger.log_check(f"Limiting EDA dataset to {max_rows} rows.")
        # df_row_count = len(input_df)
        input_df = input_df.head(max_rows)
    else:
        logger.log_result(f"Not skipping any rows in dataset.")

    # describe_dataframe(df=input_df, logger=logger, name=f"EDA {subset} dataframe")

    feature_sets = NumFeatureSets.extract_features(
        df=input_df, logger=logger, limit_features=limit_features
    )
    inspect_dataframe(df=input_df, logger=logger, name=f"EDA {subset} dataframe")
    identify_missing_values(df=input_df, logger=logger)

    plot_num_feature_distributions(
        input_df,
        logger=logger,
        feature_ctgs=feature_sets,
        # drop_cols=["raise"],
        col_type="structural",
        experiment_dir=exp_dir / "struct_features_distributions",
        rows_per_page=4,
        cols_per_page=2,
    )
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
        # top_features=5
        limit_features=[
            "time_since_last_change",
            "loc_added",
            "loc_deleted",
            "max_func_change",
            "msg_len",
        ],
    )

    # -----------------------------------------------------------------------------
    # Outliers
    # -----------------------------------------------------------------------------

    logger.log_check("Starting the outlier phase...")

    plot_boxplots_with_outliers(
        df=input_df,
        feature_sets=feature_sets,
        logger=logger,
        experiment_dir=exp_dir / "outliers",
        drop_binaries=True,
        n_cols_per_page=2,
        n_rows_per_page=4,
        # limit_features=limit_features,
    )

    # -----------------------------------------------------------------------------
    # Target Separability
    # -----------------------------------------------------------------------------

    plot_2D_embedding_separability(
        df=input_df,
        logger=logger,
        sample_size=2000,
        experiment_dir=exp_dir / "separability",
        base_fontsize=15,
    )

    return results


def get_parser(add_help: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EDA Script for ML pipeline.", add_help=add_help
    )

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

    parser.add_argument(
        "--limit-features",
        nargs="+",
        type=str,
        required=False,
        default=[
            "loc_added",
            "loc_deleted",
            "max_func_change",
            "msg_len",
            "raise",
            "recent_churn",
            "time_since_last_change",
            "activity_per_exp",
        ],
        help="Limit EDA to specific features (space-separated list).",
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
    parser = get_parser(add_help=True)
    args = parser.parse_args()

    perform_EDA(
        logger=DEF_SCRIPT_LOGGER,
        subset=args.subset,
        load_ETL_processed=args.load_etl_df,
        experiment_id=3,
        max_rows=args.max_rows,
        intersect_with_processed=args.intersect_with_processed,
        limit_features=args.limit_features,
    )
