from collections import Counter
from typing import List, get_args

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from notebooks.constants import (
    ENGINEERED_FEATURES,
    INTERACTION_FEATURES,
    LINE_TOKEN_FEATURES,
    STATISTICAL_METRICS,
    STRUCTURAL_METRICS,
    TARGET,
)
from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.experimenting.utils import log_experiment_id
from src_code.ml_pipeline.feature_importance import PFIWrapper
from src_code.ml_pipeline.models import ModelWrapperFactory, RFWrapper, XGBWrapper

from src_code.ml_pipeline.preprocessing.data_engineering import (
    aggr_line_token_features,
    create_buckets,
    create_derived_features,
    create_feature_interactions,
)
from src_code.ml_pipeline.preprocessing.transform import (
    build_transformer,
    pca_explained_variance,
    transform,
)

# from src_code.ml_pipeline.testing.testing import display_ROC_curve, evaluate, find_best_threshold, find_optimal_threshold_MCC, infer, prec_recall_curve
from src_code.ml_pipeline.training.constants import DEF_TOP_K
from src_code.ml_pipeline.training.training import (
    check_single_infer,
    fit_model,
    fit_rf,
    split_train_test,
)
from src_code.ml_pipeline.training.tuning import (
    ModelTuningFactory,
    RFTuningWrapper,
    XGBTuningWrapper,
)
from src_code.ml_pipeline.training.utils import analyze_features
from src_code.ml_pipeline.validations import CVWrapper
from src_code.mlops_intstrex.reporters.console_reporter import ConsoleReporter
from src_code.mlops_intstrex.reporters.tqdm_reporter import TqdmReporter
from src_code.utils.utils import timeit
from src_code.versioning import VersionedFileManager
from ..preprocessing.preprocessing import drop_cols, drop_invalid_rows
from ..data_utils import (
    PipelineArtifact,
    load_artifact,
    load_df,
    load_model,
    save_artifact,
    save_model,
)
from ...config import (
    ENGINEERED_DATA_DIR,
    # ENGINEERING_MAPPINGS,
    LOG_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    TUNED_DIR,
    SupportedModel,
    SupportedModel,
)
from argparse import ArgumentParser
from ..preprocessing import feature_config as ftr_cfg

RANDOM_STATE = 42
TEST_SPLIT = 0.2
PIPELINE_PHASES = ["preprocess", "train", "eval"]

# SCRIPT_LOGGER = DEF_NOTEBOOK_LOGGER
TOP_K_IMPORTANCES = 15
REFINEMENT_THRESHOLD = 0.0001
CUSTOM_THRESHOLD = 0.75

DEF_SCRIPT_LOGGER = MyLogger(
    label="TRAIN",
    section_name="TRAINING SCRIPT",
    file_log_path=LOG_DIR / "training_script.log",
)


@timeit("Training Phase", logger_param="script_logger")
def train(
    model_type: SupportedModel,
    logger: MyLogger = DEF_SCRIPT_LOGGER,
    load_tuned: bool = True,
    skip_pfi: bool = False,
    top_k: int = DEF_TOP_K,
    experiment_id: int = None,
):
    logger.start_session(
        session_id=experiment_id if experiment_id else MyLogger.DEF_SESSION_ID
    )
    log_experiment_id(logger=logger, experiment_id=experiment_id)
    model_output_file = VersionedFileManager(
        MODEL_DIR / f"{model_type.upper()}_model_train.joblib", logger=logger
    )
    logger.log_result(
        f"Config: [{model_type=}, {load_tuned=}, {skip_pfi=}, {top_k=}, {experiment_id=}]"
    )

    target_df_versioner = VersionedFileManager(
        file_path=ENGINEERED_DATA_DIR / "train_engineered.feather", logger=logger
    )
    # target_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS['train']["output"]
    target_df = load_df(target_df_versioner.current_newest)

    validate_df_versioner = VersionedFileManager(
        file_path=ENGINEERED_DATA_DIR / "val_engineered.feather", logger=logger
    )

    # validate_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS['validate']["output"]
    validate_df = load_df(validate_df_versioner.current_newest)

    # SELECTED_SUBSETS = [STATISTICAL_METRICS]
    FEATURE_SUBSETS = {
        "STATISTICAL_METRICS": STATISTICAL_METRICS,
        "STRUCTURAL_METRICS": STRUCTURAL_METRICS,
        # "STRUCTURAL_METRICS": STRUCTURAL_METRICS,
        # "SEMANTIC_METRICS": SEMANTIC_METRICS,
    }

    # SELECTED_SUBSETS = ["STATISTICAL_METRICS", "STRUCTURAL_METRICS"]
    SELECTED_SUBSETS = []
    selected_features = []

    for subset_name in SELECTED_SUBSETS:
        subset = FEATURE_SUBSETS[subset_name]
        selected_features.extend(subset)
        logger.log_result(
            f"Using feature subset '{subset_name}' ({len(subset)} features)"
        )

    if selected_features:
        selected_features.append(TARGET)
        logger.log_result(f"Total selected features: {len(selected_features)}")
        logger.log_result(f"First 5 features: {selected_features[:5]}")

        target_df = target_df[selected_features]
        validate_df = validate_df[selected_features]

    # -----------------------------------------------------------------------------
    # Dropping invalid cols
    # -----------------------------------------------------------------------------

    target_df = drop_cols(df=target_df, cols=ftr_cfg.DROP_COLS, logger=logger)
    validate_df = drop_cols(df=validate_df, cols=ftr_cfg.DROP_COLS, logger=logger)

    # -----------------------------------------------------------------------------
    # Dropping cols
    # -----------------------------------------------------------------------------

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

    object_cols = X_test.select_dtypes(include=["object"]).columns
    print(object_cols)
    tuned_hyperparams = None

    if load_tuned:
        # If loading tuned model, we override the current model's parameters

        # tuned_model_versioner = VersionedFileManager(
        #     file_path=TUNED_DIR / f"{model_type}_model_tuned.pkl", logger=logger
        # )

        try:
            tuned_hyperparams = load_artifact(
                dir=TUNED_DIR,
                artifact_type="tuning-hyperparams",
                logger=logger,
                label=model_type,
            )
            # tuned_model, features = load_model(
            #     path=tuned_model_versioner.current_newest, logger=logger
            # )
            # artifact = load_artifact()
            # model_wrapper.set_model(tuned_model)
            logger.log_check(
                f"Configuring model with new hyperparams ({len(tuned_hyperparams.hyperparams)})"
            )
            # model_wrapper.configure(tuned_hyperparams.strip_prefix(prefix="model"))
        except FileNotFoundError:
            msg = "Tuned model not found. Please run hyperparameter tuning phase before training."
            logger.logger.error(msg)
            raise FileNotFoundError(msg)

    # -----------------------------------------------------------------------------
    # Model & TrainingPipeline Definition
    # -----------------------------------------------------------------------------

    model_wrapper = ModelWrapperFactory.create(
        model_type=model_type,
        random_state=RANDOM_STATE,
        logger=logger,
        scale_pos_weight=XGBWrapper.calc_scale_pos_weight(y_train),
        top_k=top_k,
        tuned_hyperparams=tuned_hyperparams.extract_features() if tuned_hyperparams else None,
    )

    object_cols = X_test.select_dtypes(include=["object"]).columns
    # print(object_cols)

    # -----------------------------------------------------------------------------
    # Model Fit
    # -----------------------------------------------------------------------------

    # This step trains the single, final model pipeline that is saved
    # in the 'model' variable and used for prediction and PFI.
    model_wrapper = fit_model(
        model_type=model_type.upper(),
        model_wrapper=model_wrapper,
        X_train=X_train,
        y_train=y_train,
        X_validate=X_validate,
        y_validate=y_validate,
    )

    # -----------------------------------------------------------------------------
    # Single inference check
    # -----------------------------------------------------------------------------

    check_single_infer(model=model_wrapper.pipeline, X_test=X_test)

    # -----------------------------------------------------------------------------
    # PFI & Training Subset Refinement
    # -----------------------------------------------------------------------------

    if not skip_pfi:
        pfi_wrapper = PFIWrapper(
            model=model_wrapper.pipeline,
            random_state=RANDOM_STATE,
            logger=logger,
            reporter_cls=ConsoleReporter,
        )

        importances = pfi_wrapper.run_PFI(
            X_test=X_validate, y_test=y_validate, top_k=TOP_K_IMPORTANCES
        )

        X_train, X_test, X_validate = pfi_wrapper.refine_features(
            X_train=X_train,
            X_test=X_test,
            X_val=X_validate,
            threshold=REFINEMENT_THRESHOLD,
        )

        # -----------------------------------------------------------------------------
        # Model retraining
        # -----------------------------------------------------------------------------

        model_wrapper = fit_model(
            model_type=model_type.upper(),
            model_wrapper=model_wrapper,
            X_train=X_train,
            y_train=y_train,
            X_validate=X_validate,
            y_validate=y_validate,
        )

    else:
        logger.log_result("Skipping PFI process...")

    logger.log_result("Training phase finished.")

    # save_df(df=target_df, df_fil~e_path=)
    wrapper_artifact = PipelineArtifact(
        artifact_type="trained_model", label=model_type, model_wrapper=model_wrapper
    )

    save_artifact(dir=MODEL_DIR, artifact=wrapper_artifact, logger=logger)
    return model_output_file.next_base_output


def get_parser():
    parser = ArgumentParser(
        description="Parametric ML pipeline script.", add_help=False
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        choices=get_args(SupportedModel),
        default="RF",
        help="Model type to use in the pipeline.",
    )
    parser.add_argument(
        "--load-tuned",
        action="store_true",
        required=False,
        default=False,
        help="Whether to load a pre-tuned model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEF_TOP_K,
        required=False,
        help="Keep only top k features for training",
    )
    # parser.add_argument(
    #     "--skip-cv",
    #     action="store_true",
    #     required=False,
    #     default=False,
    #     help="Cross-validation is skipped in the training phase.",
    # )
    # parser.add_argument(
    #     "--skip-tuning",
    #     action="store_true",
    #     required=False,
    #     default=False,
    #     help="Hyperparameter Tunining is skipped in the training phase.",
    # )
    parser.add_argument(
        "--skip-pfi",
        action="store_true",
        required=False,
        default=False,
        help="PFI is skipped in training phase.",
    )

    return parser


if __name__ == "__main__":
    script_logger = DEF_SCRIPT_LOGGER
    # script_logger.start_session()

    parser = get_parser()

    args = parser.parse_args()
    # filtered_phases: List[str] = args.phases
    # subset: SubsetType = args.subset
    # MODEL_TYPE = args.model  # "rf" or "xgb"
    train(
        model_type=args.model,
        logger=script_logger,
        load_tuned=args.load_tuned,
        skip_pfi=args.skip_pfi,
        experiment_id=None,
    )
