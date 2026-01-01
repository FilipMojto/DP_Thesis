from typing import List, get_args
from notebooks.constants import (
    ENGINEERED_FEATURES,
    INTERACTION_FEATURES,
    LINE_TOKEN_FEATURES,
    TARGET,
)
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
from src_code.ml_pipeline.testing.testing import display_ROC_curve, evaluate, find_best_threshold, find_optimal_threshold_MCC, infer, prec_recall_curve
from src_code.ml_pipeline.training.train import (
    check_single_infer,
    fit_model,
    fit_rf,
    split_train_test,
)
from src_code.ml_pipeline.training.tuning import ModelTuningFactory, RFTuningWrapper, XGBTuningWrapper
from src_code.ml_pipeline.training.utils import analyze_features
from src_code.ml_pipeline.validations import CVWrapper
from .preprocessing.preprocessing import drop_invalid_rows
from .data_utils import load_df, load_model, save_model
from ..config import ENGINEERING_MAPPINGS, MODEL_DIR, SupportedModels, SupportedModels
from argparse import ArgumentParser
from .preprocessing import feature_config as ftr_cfg

RANDOM_STATE = 42
TEST_SPLIT = 0.2
PIPELINE_PHASES = ["preprocess", "train", "eval"]

SCRIPT_LOGGER = DEF_NOTEBOOK_LOGGER
TOP_K_IMPORTANCES = 15
REFINEMENT_THRESHOLD = 0.0001
CUSTOM_THRESHOLD = 0.75


if __name__ == "__main__":
    SCRIPT_LOGGER.log_check("[script] starting ML script...")
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
    script_model_path = MODEL_DIR / f"{MODEL_TYPE.upper()}_model_script_train.joblib"   

    # =============================================================================
    # TRAINING
    # =============================================================================

    # -----------------------------------------------------------------------------
    # Loading df
    # -----------------------------------------------------------------------------
    target_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS['train']["output"]
    target_df = load_df(target_df_path)

    validate_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS['validate']["output"]
    validate_df = load_df(validate_df_path)

    SCRIPT_LOGGER.log_check("Starting training phase...")

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

  
    model_wrapper, step_name = ModelWrapperFactory.create(model_type=MODEL_TYPE.lower(), random_state=RANDOM_STATE)
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
            y_train=y_train
        )

        best_params, best_score = tuning.run_grid_search()
        model.set_params(**best_params)
    else:
        SCRIPT_LOGGER.log_result("Skipping Hyperparameter Tuning...")

    # -----------------------------------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------------------------------

    if not args.skip_cv:
        cv_wrapper = CVWrapper(random_state=RANDOM_STATE)
   
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
        SCRIPT_LOGGER.log_result("Skipping cross-validation...")

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
            model=model, random_state=RANDOM_STATE, logger=SCRIPT_LOGGER
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
        SCRIPT_LOGGER.log_result("Skipping PFI process...")

    SCRIPT_LOGGER.log_result("Training phase finished.")

    # save_df(df=target_df, df_fil~e_path=)
    save_model(
        model=model,
        path=script_model_path
    )

    # # if not filtered_phases or "eval" in filtered_phases:
    # # =============================================================================
    # # FINAL EVALUATION
    # # =============================================================================
    
    # # -----------------------------------------------------------------------------
    # # Dataset Loading
    # # -----------------------------------------------------------------------------
    # target_df_path = TARGET_DF_FILE = ENGINEERING_MAPPINGS['test']["output"]
    # test_df = load_df(df_file_path=target_df_path)
    

    # # -----------------------------------------------------------------------------
    # # Model Loading
    # # -----------------------------------------------------------------------------

    # model_wrapper = load_model(path=script_model_path)
    # model_features = model_wrapper.feature_names_in_

    # # -----------------------------------------------------------------------------
    # # Column Filtering
    # # -----------------------------------------------------------------------------

    # X_test = test_df[model_features]

    # # -----------------------------------------------------------------------------
    # # Inference
    # # -----------------------------------------------------------------------------
    
    # y_true = test_df["label"] if "label" in test_df.columns else None

    # predictions, probabilities = infer(X_test=X_test, model=model_wrapper)

    # # -----------------------------------------------------------------------------
    # # Evaluation
    # # -----------------------------------------------------------------------------
    
    # evaluate(y_true=y_true, predictions=predictions, probabilities=probabilities)

    # # -----------------------------------------------------------------------------
    # # Precision-Recall Curve
    # # -----------------------------------------------------------------------------
    
    # precision, recall, thresholds = prec_recall_curve(y_true=y_true, probs=probabilities)


    # # -----------------------------------------------------------------------------
    # # Optimal Threshold for MCC
    # # -----------------------------------------------------------------------------
    
    # best_mcc_threshold, best_mcc = find_optimal_threshold_MCC(y_true=y_true, probs=probabilities)

    # # 4. Generate the final report
    # # final_predictions = (probs >= best_threshold).astype(int)
    # # print(classification_report(y_true, final_predictions))
    # evaluate(
    #     y_true=y_true,
    #     predictions=predictions,
    #     probabilities=probabilities,
    #     threshold=best_mcc_threshold,
    # )

    # # -----------------------------------------------------------------------------
    # # ROC Curve
    # # -----------------------------------------------------------------------------
    
    # display_ROC_curve(y_true=y_true, probabilities=probabilities)

