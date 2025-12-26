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
TEST_SPLIT = 0.2
PIPELINE_PHASES = ["preprocess", "train", "eval"]

SCRIPT_LOGGER = DEF_NOTEBOOK_LOGGER
TOP_K_IMPORTANCES = 15
REFINEMENT_THRESHOLD = 0.0001

if __name__ == "__main__":
    SCRIPT_LOGGER.log_check("[script] starting ML script...")
    parser = ArgumentParser(description="Parametric ML pipeline script.")
    parser.add_argument(
        "--subset",
        choices=get_args(SubsetType),
        default="train",
        required=False,
        help="Specify which subset (train, test or validate) to run through the pipeline.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=PIPELINE_PHASES,
        default=[],
        help="Pipeline phases to run",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        required=False,
        default=False,
        help="Cross-validation is skipped in the training phase.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        required=False,
        default=False,
        help="Hyperparameter Tunining is skipped in the training phase.",
    )
    parser.add_argument(
        "--skip-pfi",
        action="store_true",
        required=False,
        default=False,
        help="PFI is skipped in training phase.",
    )

    args = parser.parse_args()
    filtered_phases: List[str] = args.phases
    subset: SubsetType = args.subset

    if not isinstance(filtered_phases, list) or (
        filtered_phases
        and not all(phase in PIPELINE_PHASES for phase in filtered_phases)
    ):
        err_msg = f"[SCRIPT] Invalid argument (phases) value: {filtered_phases}"
        SCRIPT_LOGGER.logger.error(
            f"Invalid argument (phases) value: {filtered_phases}"
        )
        raise ValueError(err_msg)

    # subset: SubsetType = "train"

    if not filtered_phases or "preprocess" in filtered_phases:
        SCRIPT_LOGGER.log_check("Starting preprocessing phase...")

        # =============================================================================
        # PREPROCESSING
        # =============================================================================

        # target_df_path = TARGET_DF_FILE = PREPROCESSING_MAPPINGS[subset]["input"]

        target_df_path = TARGET_DF_FILE = PREPROCESSING_MAPPINGS[subset]["input"]
        target_df = load_df(target_df_path)
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

        SCRIPT_LOGGER.log_result(
            f"Message embeddings explain "
            f"{pca_explained_variance(fitted_transformer, 'msg_embed'):.2%} of variance"
        )

        # -----------------------------------------------------------------------------
        # Data Engineering
        # -----------------------------------------------------------------------------
        before_engineer_cols = set(target_df.columns)

        SCRIPT_LOGGER.log_check("Starting data engineering subphase...")
        # -----------------------------------------------------------------------------
        # Feature Derivation
        # -----------------------------------------------------------------------------

        # mappings = {
        #     "loc_churn_ratio": lambda df: df["loc_added"] / (df["loc_deleted"] + 1),
        #     "activity_per_exp": lambda df: df["author_recent_activity_pre"]
        #     / (df["author_exp_pre"] + 1),
        # }

        # [STAGE 1] Derived Features
        target_df = create_derived_features(
            df=target_df, mappings=ftr_cfg.DERIVED_FEATURES
        )
        # [STAGE 2] Creating Buckets
        target_df = create_buckets(
            df=target_df, mappings=ftr_cfg.BUCKET_MAPPINGS, encode=True
        )
        # [STAGE 3] Aggregating line token features
        target_df = aggr_line_token_features(df=target_df, features=LINE_TOKEN_FEATURES)
        # [STAGE 4] Feature interactions
        target_df = create_feature_interactions(
            df=target_df, features=INTERACTION_FEATURES
        )

        SCRIPT_LOGGER.log_result("Data engineering subphase finished.")
        # SCRIPT_LOGGER.log_result(f"Engineered features: {ENGINEERED_FEATURES}", print_to_console=True)
        after_engineer_cols = set(target_df.columns)
        SCRIPT_LOGGER.log_result(
            f"Engineered features: {after_engineer_cols - before_engineer_cols}",
            print_to_console=True,
        )

        SCRIPT_LOGGER.log_result("Preprocessing phase finished.")

        save_df(df=target_df, df_file_path=ENGINEERING_MAPPINGS[subset]["output"])

    if not filtered_phases or "train" in filtered_phases:
        # =============================================================================
        # TRAINING
        # =============================================================================

        # -----------------------------------------------------------------------------
        # Loading df
        # -----------------------------------------------------------------------------
        target_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS[subset]["output"]
        target_df = load_df(target_df_path)

        SCRIPT_LOGGER.log_check("Starting training phase...")

        # -----------------------------------------------------------------------------
        # Dropping cols
        # -----------------------------------------------------------------------------

        target_df = drop_cols(df=target_df, cols=ftr_cfg.DROP_COLS)

        # -----------------------------------------------------------------------------
        # analyzigin features
        # -----------------------------------------------------------------------------

        analyze_features(df=target_df, target=TARGET)

        # -----------------------------------------------------------------------------
        # Traing&Test Split
        # -----------------------------------------------------------------------------

        X_train, X_test, y_train, y_test = split_train_test(
            df=target_df, target=TARGET, random_state=RANDOM_STATE, test_size=TEST_SPLIT
        )

        # -----------------------------------------------------------------------------
        # Model & TrainingPipeline Definition
        # -----------------------------------------------------------------------------

        rf_wrapper = RFWrapper(random_state=RANDOM_STATE)
        model = rf_wrapper.get_model()

        rf_pipeline = RFPipelineWrapper(rf=model)
        # model = rf_wrapper.get_model()

        # -----------------------------------------------------------------------------
        # Hyperparameter Tuning
        # -----------------------------------------------------------------------------
        if not args.skip_tuning:
            rf_tuning_wrapper = RFTuningWrapper(rf=model, X_train=X_train, y_train=y_train)
            best_params, best_score = rf_tuning_wrapper.run_grid_search()

            # --- Update Model ---
            model.set_params(**best_params)  # make sure pipeline/model uses the tuned parameters
        else:
            SCRIPT_LOGGER.log_result("Skipping Hyperparameter Tuning...")

        # -----------------------------------------------------------------------------
        # Cross-validation
        # -----------------------------------------------------------------------------

        if not args.skip_cv:
            cv_wrapper = CVWrapper(random_state=RANDOM_STATE)
            cv_results = cv_wrapper.cross_validate(
                model=model, X_train=X_train, y_train=y_train
            )

            cv_wrapper.mean_results()
        else:
            SCRIPT_LOGGER.log_result("Skipping cross-validation...")

        # -----------------------------------------------------------------------------
        # Model Fit
        # -----------------------------------------------------------------------------

        # This step trains the single, final model pipeline that is saved
        # in the 'model' variable and used for prediction and PFI.
        # model.fit(X_train, y_train)
        model = fit_model(model=model, X_train=X_train, y_train=y_train)

        # -----------------------------------------------------------------------------
        # Single inference check
        # -----------------------------------------------------------------------------

        check_single_infer(model=model, X_test=X_test)

        # -----------------------------------------------------------------------------
        # PFI & Training Subset Refinement
        # -----------------------------------------------------------------------------

        if not args.skip_pfi:
            pfi_wrapper = PFIWrapper(
                model=model, X_test=X_test, y_test=y_test, random_state=RANDOM_STATE
            )
            pfi_wrapper.run_PFI()
            pfi_wrapper.calc_importances()
            importances = pfi_wrapper.get_importances(top_k=TOP_K_IMPORTANCES)

            X_train, X_test = pfi_wrapper.refine_features(
                X_train=X_train, X_test=X_test, threshold=REFINEMENT_THRESHOLD
            )

            # -----------------------------------------------------------------------------
            # Model retraining
            # -----------------------------------------------------------------------------
            # model.fit(X_train, y_train)
            model = fit_model(model=model, X_train=X_train, y_train=y_train)
        else:
            SCRIPT_LOGGER.log_result("Skipping PFI process...")

        SCRIPT_LOGGER.log_result("Training phase finished.")

        # save_df(df=target_df, df_fil~e_path=)
        save_model(model=model, path=MODEL_DIR / 'RF_model_script.joblib')


    if not filtered_phases or "eval" in filtered_phases:
        # NOTE: TODO
        ...
