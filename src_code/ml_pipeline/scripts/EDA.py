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
from src_code.ml_pipeline.EDA.plots import (
    plot_categorical_comparison,
    plot_num_feature_distributions,
)
from src_code.ml_pipeline.EDA.utils import extract_features
from src_code.ml_pipeline.data_utils import load_df
from src_code.ml_pipeline.experimenting.utils import log_experiment_id
from src_code.ml_pipeline.utils import describe_dataframe, get_experiment_dir
from src_code.versioning import VersionedFileManager


DEF_SCRIPT_LOGGER = MyLogger(
    label="EDA", section_name="EDA", file_log_path=LOG_DIR / "EDA.log"
)


def perform_EDA(
    logger: MyLogger,
    subset: SubsetType = "train",
    load_ETL_processed: bool = False,
    experiment_id: int = None,
    max_rows: int = None,
):
    if logger == DEF_SCRIPT_LOGGER:
        # If default logger is used, start a new session
        logger.start_session()

    log_experiment_id(logger=logger, experiment_id=experiment_id)
    logger.log_check(f"etl_processed: {load_ETL_processed}")

    # input_df_path = (
    #     EXTENDED_DATA_DIR / f"{subset}_extended.feather"
    #     if load_ETL_processed
    #     else PROCESSED_DATA_DIR / f"{subset}_transformed.feather"
    # )
    # input_df_versioner = VersionedFileManager(file_path=input_df_path, logger=logger)
    # input_df = load_df(df_file_path=input_df_versioner.current_newest, logger=logger)

    df_labels: List[SubsetType] = ["train", "test", "val"]
    dfs: Dict[str, pd.DataFrame] = {}

    for df_label in df_labels:
        df_path = (
            EXTENDED_DATA_DIR / f"{df_label}_extended.feather"
            if load_ETL_processed
            else PROCESSED_DATA_DIR / f"{df_label}_transformed.feather"
        )
        df_versioner = VersionedFileManager(file_path=df_path, logger=logger)
        dfs[df_label] = load_df(df_file_path=df_versioner.current_newest, logger=logger)

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

    feature_ctgs = extract_features(df=input_df, logger=logger)

    plot_num_feature_distributions(
        input_df,
        logger=logger,
        feature_ctgs=feature_ctgs,
        drop_cols=["raise"],
        col_type="structural",
        experiment_dir=exp_dir / "struct_features_distributions",
        rows_per_page=5,
    )
    plot_num_feature_distributions(
        input_df,
        logger=logger,
        feature_ctgs=feature_ctgs,
        col_type="engineered",
        experiment_dir=exp_dir / "struct_features_engineered_distributions",
        rows_per_page=5,
    )

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

    plot_categorical_comparison(
        dfs=dfs,
        feature=TARGET,
        logger=logger,
        experiment_dir=exp_dir / "other_distributions",
        rows_per_page=3,
    )

    plot_categorical_comparison(
        dfs=dfs,
        feature='repo',
        logger=logger,
        experiment_dir=exp_dir / "other_distributions",
        rows_per_page=3,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="EDA Script for ML pipeline.")

    argparser.add_argument(
        "--subset",
        choices=get_args(SubsetType),
        default="train",
        required=False,
        help="Specify which subset (train, test or validate) to run through the pipeline.",
    )

    argparser.add_argument(
        "--load-etl-df",
        action="store_true",
        required=False,
        default=False,
        help="Loads and uses ETL-processed dataset istead of classic pipeline-processed one.",
    )

    argparser.add_argument(
        "--max-rows",
        type=int,
        required=False,
        default=None,
        help="Limit dataset to first n rows only for testing purposes.",
    )

    args = argparser.parse_args()

    perform_EDA(
        logger=DEF_SCRIPT_LOGGER,
        subset=args.subset,
        load_ETL_processed=args.load_etl_df,
        experiment_id=4,
        max_rows=args.max_rows,
    )
