import argparse
from typing import get_args
from notebooks.logging_config import MyLogger
from src_code.config import EDA_DIR, EXTENDED_DATA_DIR, LOG_DIR, PROCESSED_DATA_DIR, SubsetType
from src_code.ml_pipeline.EDA.plots import plot_embedding_distributions, plot_num_feature_distributions
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
):
    if logger == DEF_SCRIPT_LOGGER:
        # If default logger is used, start a new session
        logger.start_session()

    log_experiment_id(logger=logger, experiment_id=experiment_id)
    logger.log_check(f"etl_processed: {load_ETL_processed}")

    input_df_path = (
        EXTENDED_DATA_DIR / f"{subset}_extended.feather"
        if load_ETL_processed
        else PROCESSED_DATA_DIR / f"{subset}_transformed.feather"
    )
    input_df_versioner = VersionedFileManager(file_path=input_df_path, logger=logger)
    input_df = load_df(df_file_path=input_df_versioner.current_newest, logger=logger)
    # experiment_dir = EDA_DIR if experiment_id else None
    exp_dir = get_experiment_dir(experiment_id, target_dir=EDA_DIR) if experiment_id else None


    describe_dataframe(df=input_df, logger=logger, name=f"EDA {subset} dataframe")

    feature_ctgs = extract_features(df=input_df, logger=logger)

    plot_num_feature_distributions(input_df, logger=logger, feature_ctgs=feature_ctgs, col_type='structural', experiment_dir=exp_dir / "struct_features_distributions")
    
    # plot_num_feature_distributions(input_df, logger=logger, feature_ctgs=feature_ctgs, cols=feature_ctgs.embedding_cols, experiment_dir=exp_dir / "embedding_distributions")
    # plot_num_feature_distributions(input_df, logger=logger, feature_ctgs=feature_ctgs, cols=feature_ctgs.tfidf_vectorized_cols, experiment_dir=exp_dir / "tfidf_vectorized_distributions")
    
    if not load_ETL_processed:
        logger.log_check("Plotting embeddings and vectors...")
        # plot_embedding_distributions(df=input_df, feature_ctgs=feature_ctgs, logger=logger, experiment_dir=exp_dir / "embedding_distributions")
        plot_num_feature_distributions(input_df, logger=logger, feature_ctgs=feature_ctgs, col_type='embedding', experiment_dir=exp_dir / "embedding_distributions")
        plot_num_feature_distributions(input_df, logger=logger, feature_ctgs=feature_ctgs, col_type='vectorized', experiment_dir=exp_dir / "tfidf_vectorized_distributions")

    else:
        logger.log_result("ETL-processed data does not have embeddings!")


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

    args = argparser.parse_args()

    perform_EDA(
        logger=DEF_SCRIPT_LOGGER,
        subset=args.subset,
        load_ETL_processed=args.load_etl_df,
        experiment_id=4
    )
