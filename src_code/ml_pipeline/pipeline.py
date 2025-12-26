from typing import List, get_args
from notebooks.constants import (
    ENGINEERED_FEATURES,
    INTERACTION_FEATURES,
    LINE_TOKEN_FEATURES,
    TARGET,
)
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.feature_importance import PFIWrapper
from src_code.ml_pipeline.models import RFWrapper
from src_code.ml_pipeline.pipelines import RFPipelineWrapper
from src_code.ml_pipeline.preprocessing.data_engineering import (
    aggr_line_token_features,
    create_buckets,
    create_derived_features,
    create_feature_interactions,
)
from src_code.ml_pipeline.preprocessing.transform import (
    pca_explained_variance,
    transform,
)
from src_code.ml_pipeline.training.train import (
    check_single_infer,
    fit_model,
    split_train_test,
)
from src_code.ml_pipeline.training.tuning import RFTuningWrapper
from src_code.ml_pipeline.training.utils import analyze_features, drop_cols
from src_code.ml_pipeline.validations import CVWrapper
from .preprocessing.preprocessing import drop_invalid_rows
from .df_utils import load_df, save_df, save_model
from ..config import ENGINEERING_MAPPINGS, MODEL_DIR, PREPROCESSING_MAPPINGS, SubsetType
from argparse import ArgumentParser
from . import feature_config as ftr_cfg

RANDOM_STATE = 42

if __name__ == "__main__":
    subset: SubsetType = "train"
    target_df_path = TARGET_DF_FILE = PREPROCESSING_MAPPINGS[subset]["input"]


    print("HERE")
    target_df = load_df(target_df_path)
    target_df = drop_invalid_rows(
        df=target_df,
        # numeric_features=NUMERIC_FEATURES,
        # row_filters={"time_since_last_change": target_df["time_since_last_change"] < 0},
        row_filters={"time_since_last_change": lambda s: s>= 0}
    )

        # -----------------------------------------------------------------------------
        # Transformations
        # -----------------------------------------------------------------------------

        target_df, fitted_transformer = transform(
            df=target_df,
            subset=subset,
            random_state=RANDOM_STATE,
        )

        # --- Variance Explanation by Embeddings - Demo ---

        SCRIPT_LOGGER.log_result(
            f"Code embeddings explain "
            f"{pca_explained_variance(fitted_transformer, 'code_embed'):.2%} of variance"
        )

    print(
        f"Message embeddings explain "
        f"{fitted_transformer.pca_explained_variance('msg_embed'):.2%} of variance"
    )
