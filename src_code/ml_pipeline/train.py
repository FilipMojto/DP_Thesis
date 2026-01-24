from collections import Counter
from typing import List, get_args
from notebooks.constants import (
    ENGINEERED_FEATURES,
    INTERACTION_FEATURES,
    LINE_TOKEN_FEATURES,
    TARGET,
)
from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.feature_importance import PFIWrapper
from src_code.ml_pipeline.models import ModelWrapperFactory, RFWrapper, XGBWrapper
from src_code.ml_pipeline.pipelines import ModelPipelineWrapper
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
# from src_code.ml_pipeline.testing.testing import display_ROC_curve, evaluate, find_best_threshold, find_optimal_threshold_MCC, infer, prec_recall_curve
from src_code.ml_pipeline.training.train import (
    check_single_infer,
    fit_model,
    fit_rf,
    split_train_test,
)
from src_code.ml_pipeline.training.tuning import ModelTuningFactory, RFTuningWrapper, XGBTuningWrapper
from src_code.ml_pipeline.training.utils import analyze_features
from src_code.ml_pipeline.validations import CVWrapper
from src_code.versioning import VersionedFileManager
from .preprocessing.preprocessing import drop_invalid_rows
from .data_utils import load_df, load_model, save_model
from ..config import ENGINEERING_MAPPINGS, LOG_DIR, MODEL_DIR, PROCESSED_DATA_DIR, SupportedModels, SupportedModels
from argparse import ArgumentParser
from .preprocessing import feature_config as ftr_cfg

RANDOM_STATE = 42
TEST_SPLIT = 0.2
PIPELINE_PHASES = ["preprocess", "train", "eval"]

# SCRIPT_LOGGER = DEF_NOTEBOOK_LOGGER
TOP_K_IMPORTANCES = 15
REFINEMENT_THRESHOLD = 0.0001
CUSTOM_THRESHOLD = 0.75


if __name__ == "__main__":
    script_logger = MyLogger(label="TRAIN", section_name="TRAINING SCRIPT", file_log_path=LOG_DIR / "training_script.log")
    script_logger.start_session()
    # script_logger.log_check("Starting training script...")
    parser = ArgumentParser(description="Parametric ML pipeline script.")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        choices=get_args(SupportedModels),
        default="RF",
        help="Model type to use in the pipeline.",
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
    # filtered_phases: List[str] = args.phases
    # subset: SubsetType = args.subset
    MODEL_TYPE = args.model  # "rf" or "xgb"

    # script_model_path = MODEL_DIR / f"RF_model_script_train.joblib"
    # script_model_path = MODEL_DIR / f"{MODEL_TYPE.upper()}_model_script_train.joblib"   
    model_file = VersionedFileManager(MODEL_DIR / f"{MODEL_TYPE.upper()}_model_train.joblib")

    RESTRICTED_COLS = {'loc_deleted': 0.36989620957290453, 'hunks_count': 0.32891914023397506, 'loc_added': 0.27842754912252743, 'files_changed': 0.2765063273157114, 'ast_delta': 0.25869381693522975, 'max_func_change': 0.25196385880821387, 'complexity_delta': 0.2479336343509937, 'msg_len': 0.23487740034222812}
    RESTRICTED_COLS_NAMES = [name for name, _ in RESTRICTED_COLS.items()]



    # =============================================================================
    # TRAINING
    # =============================================================================

    # -----------------------------------------------------------------------------
    # Loading df
    # -----------------------------------------------------------------------------
    target_df_versioner = VersionedFileManager(file_path=PROCESSED_DATA_DIR / "train_engineered.feather")
    # target_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS['train']["output"]
    target_df = load_df(target_df_versioner.current_newest)

    validate_df_versioner = VersionedFileManager(file_path=PROCESSED_DATA_DIR / "val_engineered.feather")

    # validate_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS['validate']["output"]
    validate_df = load_df(validate_df_versioner.current_newest)

    # Only keep restricted features + target
    selected_features = RESTRICTED_COLS_NAMES + [TARGET]

    # df_reduced = df[selected_features].copy()
    # target_df = target_df[selected_features]
    # validate_df = validate_df[selected_features]



    script_logger.log_check("Starting training phase...")

    # -----------------------------------------------------------------------------
    # Dropping cols
    # -----------------------------------------------------------------------------

    # target_df = drop_cols(df=target_df, cols=ftr_cfg.DROP_COLS)

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
    X_validate = validate_df.drop(columns=[TARGET])
    y_validate = validate_df[TARGET]
    

    # -----------------------------------------------------------------------------
    # Model & TrainingPipeline Definition
    # -----------------------------------------------------------------------------

    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]  # weight = #negatives / #positives

  
    model_wrapper, step_name = ModelWrapperFactory.create(model_type=MODEL_TYPE.lower(), random_state=RANDOM_STATE, logger=script_logger, scale_pos_weight=scale_pos_weight)
    model = model_wrapper.get_model()

    # rf_pipeline = ModelPipelineWrapper(rf=model)
    pipeline_wrapper = ModelPipelineWrapper(
        model=model,
        step_name=step_name
    )
    pipeline = pipeline_wrapper.get_pipeline()
    # model = rf_wrapper.get_model()

    # -----------------------------------------------------------------------------
    # Hyperparameter Tuning
    # -----------------------------------------------------------------------------
    if not args.skip_tuning:
        tuning = ModelTuningFactory.create(
            model_type=MODEL_TYPE.lower(),
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_validate,
            y_val=y_validate,
            logger=script_logger
        )

        best_params, best_score = tuning.run_grid_search()
        model.set_params(**best_params)
    else:
        script_logger.log_result("Skipping Hyperparameter Tuning...")

    # -----------------------------------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------------------------------

    if not args.skip_cv:
        cv_wrapper = CVWrapper(random_state=RANDOM_STATE, logger=script_logger)
   
        # We pass the global validation set and the step_name (xgb or rf)
        cv_results = cv_wrapper.cross_validate(
            model=pipeline,              # This is your Pipeline object
            X_train=X_train, 
            y_train=y_train,
            X_val=X_validate,           # Your separate validation set
            y_val=y_validate,
            step_name=step_name          # 'xgb' or 'rf' from your Factory
        )

        cv_wrapper.mean_results()
    else:
        script_logger.log_result("Skipping cross-validation...")

    # -----------------------------------------------------------------------------
    # Model Fit
    # -----------------------------------------------------------------------------

    # This step trains the single, final model pipeline that is saved
    # in the 'model' variable and used for prediction and PFI.
    model_wrapper = fit_model(
        model_type=MODEL_TYPE.upper(),
        model_wrapper=model_wrapper,
        X_train=X_train,
        y_train=y_train,
        X_validate=X_validate,
        y_validate=y_validate
    )
        
    # -----------------------------------------------------------------------------
    # Single inference check
    # -----------------------------------------------------------------------------

    check_single_infer(model=model, X_test=X_test)

    # -----------------------------------------------------------------------------
    # PFI & Training Subset Refinement
    # -----------------------------------------------------------------------------

    if not args.skip_pfi:
        # pfi_wrapper = PFIWrapper(
        #     model=model, X_test=X_test, y_test=y_test, random_state=RANDOM_STATE
        # )
        pfi_wrapper = PFIWrapper(
            model=model, random_state=RANDOM_STATE, logger=script_logger
        )
        importances = pfi_wrapper.run_PFI(X_test=X_test, y_test=y_test, top_k=TOP_K_IMPORTANCES)
        # pfi_wrapper.calc_importances()
        # importances = pfi_wrapper.get_importances(top_k=TOP_K_IMPORTANCES)

        X_train, X_test, X_validate = pfi_wrapper.refine_features(
            X_train=X_train, X_test=X_test, X_val=X_validate, threshold=REFINEMENT_THRESHOLD
        )

        # -----------------------------------------------------------------------------
        # Model retraining
        # -----------------------------------------------------------------------------

        model_wrapper = fit_model(
            model_type=MODEL_TYPE.upper(),
            model_wrapper=model_wrapper,
            X_train=X_train,
            y_train=y_train,
            X_validate=X_validate,
            y_validate=y_validate
        )

    else:
        script_logger.log_result("Skipping PFI process...")

    script_logger.log_result("Training phase finished.")

    # save_df(df=target_df, df_fil~e_path=)
    save_model(
        model=model,
        # path=script_model_path
        path=model_file.next_base_output
    )


