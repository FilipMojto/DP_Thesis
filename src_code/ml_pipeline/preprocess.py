
import argparse
import time
from typing_extensions import get_args
from main_config import RANDOM_STATE
from notebooks.constants import INTERACTION_FEATURES, LINE_TOKEN_FEATURES
from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERING_MAPPINGS, EXTENDED_DATA_DIR, INTERIM_DATA_DIR, LOG_DIR, PREPROCESSING_MAPPINGS, PROCESSED_DATA_DIR, SubsetType
import src_code.ml_pipeline.data_utils as dutls
import src_code.ml_pipeline.preprocessing.preprocessing as prep
import src_code.ml_pipeline.preprocessing.data_engineering as de
import src_code.ml_pipeline.preprocessing.transform as tr
import src_code.ml_pipeline.preprocessing.feature_config as ftr_cfg
from src_code.versioning import VersionedFileManager


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(
        description="Preprocessing Script for ML Pipeline"
    )

    parser.add_argument(
        "--subset",
        choices=get_args(SubsetType),
        default="train",
        required=False,
        help="Specify which subset (train, test or validate) to run through the pipeline.",
    )

    args = parser.parse_args()
    subset: SubsetType = args.subset


    script_logger = MyLogger(label="PREPROCESS", section_name="PREPROCESS LOGGER SCRIPT", file_log_path=LOG_DIR / "preprocess_log.log")
    
    script_logger.start_session()
    script_logger.log_check("Starting preprocessing phase...")

    # =============================================================================
    # PREPROCESSING
    # =============================================================================

    # target_df_path = TARGET_DF_FILE = PREPROCESSING_MAPPINGS[subset]["input"]
    input_df_file = VersionedFileManager(file_path=EXTENDED_DATA_DIR / f"{subset}_extended")
    output_df_file = VersionedFileManager(file_path=PROCESSED_DATA_DIR / f"{subset}_engineered")

    # target_df_path = TARGET_DF_FILE = PREPROCESSING_MAPPINGS[subset]["input"]
    target_df_path = input_df_file.current_newest
    target_df = dutls.load_df(target_df_path)

    script_logger.log_result(f"Initial dataframe shape: {target_df.shape}", print_to_console=True)

    # -----------------------------------------------------------------------------
    # Dropping invalid cols
    # -----------------------------------------------------------------------------
    
    target_df = prep.drop_cols(df=target_df, cols=ftr_cfg.DROP_COLS, logger=script_logger)

    # -----------------------------------------------------------------------------
    # Dropping invalid rows
    # -----------------------------------------------------------------------------

    target_df = prep.drop_invalid_rows(
        df=target_df,
        # numeric_features=NUMERIC_FEATURES,
        # row_filters={"time_since_last_change": target_df["time_since_last_change"] < 0},
        row_filters={"time_since_last_change": lambda s: s >= 0},
    )

    # # -----------------------------------------------------------------------------
    # # Transformations
    # # -----------------------------------------------------------------------------

    # target_df, fitted_transformer = transform(
    #     df=target_df,
    #     subset=subset,
    #     random_state=RANDOM_STATE,
    # )

    # # --- Variance Explanation by Embeddings - Demo ---

    # SCRIPT_LOGGER.log_result(
    #     f"Code embeddings explain "
    #     f"{pca_explained_variance(fitted_transformer, 'code_embed'):.2%} of variance"
    # )
    # -----------------------------------------------------------------------------
    # Transformations
    # -----------------------------------------------------------------------------

    target_df, fitted_transformer = tr.transform(
        df=target_df,
        subset=subset,
        random_state=RANDOM_STATE,
    )

    # --- Variance Explanation by Embeddings - Demo ---

    script_logger.log_result(
        f"Code embeddings explain "
        f"{tr.pca_explained_variance(fitted_transformer, 'code_embed'):.2%} of variance"
    )

    script_logger.log_result(
        f"Message embeddings explain "
        f"{tr.pca_explained_variance(fitted_transformer, 'msg_embed'):.2%} of variance"
    )

    # -----------------------------------------------------------------------------
    # Data Engineering
    # -----------------------------------------------------------------------------
    before_engineer_cols = set(target_df.columns)

    script_logger.log_check("Starting data engineering subphase...")
    # -----------------------------------------------------------------------------
    # Feature Derivation
    # -----------------------------------------------------------------------------

    # mappings = {
    #     "loc_churn_ratio": lambda df: df["loc_added"] / (df["loc_deleted"] + 1),
    #     "activity_per_exp": lambda df: df["author_recent_activity_pre"]
    #     / (df["author_exp_pre"] + 1),
    # }

    # [STAGE 1] Derived Features
    target_df = de.create_derived_features(
        df=target_df, mappings=ftr_cfg.DERIVED_FEATURES
    )
    # [STAGE 2] Creating Buckets
    target_df = de.create_buckets(
        df=target_df, mappings=ftr_cfg.BUCKET_MAPPINGS, encode=True
    )
    # [STAGE 3] Aggregating line token features
    target_df = de.aggr_line_token_features(df=target_df, features=LINE_TOKEN_FEATURES)
    # [STAGE 4] Feature interactions
    target_df = de.create_feature_interactions(
        df=target_df, features=INTERACTION_FEATURES
    )

    script_logger.log_result("Data engineering subphase finished.")
    # SCRIPT_LOGGER.log_result(f"Engineered features: {ENGINEERED_FEATURES}", print_to_console=True)
    after_engineer_cols = set(target_df.columns)
    script_logger.log_result(
        f"Engineered features: {after_engineer_cols - before_engineer_cols}",
        print_to_console=True,
    )

    script_logger.log_result("Preprocessing phase finished.")
    end = time.time()
    script_logger.log_result(f"Preprocessing time: {end - start:.2f} seconds.")
    script_logger.log_result(f"Final dataframe shape: {target_df.shape}", print_to_console=True)

    # dutls.save_df(df=target_df, df_file_path=ENGINEERING_MAPPINGS[subset]["output"])
    dutls.save_df(df=target_df, df_file_path=output_df_file.next_base_output)