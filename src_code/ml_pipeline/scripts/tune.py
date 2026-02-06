from pathlib import Path
import argparse
import random

import joblib


from notebooks.constants import TARGET
from notebooks.logging_config import MyLogger
from src_code.config import (
    ENGINEERING_MAPPINGS,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    TUNED_DIR,
)
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER, DEF_RANDOM_STATE
from src_code.ml_pipeline.data_utils import load_df, save_model
from src_code.ml_pipeline.experimenting.utils import log_experiment_id
from src_code.ml_pipeline.models import ModelWrapperFactory, XGBWrapper
from src_code.config import LOG_DIR
from src_code.ml_pipeline.preprocessing.feature_config import (
    DROP_COLS,
    TUNING_DROP_COLS,
)
from src_code.ml_pipeline.preprocessing.preprocessing import drop_cols
from src_code.ml_pipeline.training.tuning import ModelTuningFactory, log_selected_features
from src_code.ml_pipeline.utils import MyParser, get_n_jobs
from src_code.utils.utils import logerror, timeit
from src_code.versioning import VersionedFileManager

DEF_SCRIPT_LOGGER = MyLogger(
    label="TUNING",
    section_name="TUNING LOGGER SCRIPT",
    file_log_path=LOG_DIR / "tuning_log.log",
)


@timeit("Hyperparameter Tuning Phase", logger_param="logger")
@logerror("Hyperparameter Tuning Phase", logger_param="logger")
def tune_hyperparams(
    preprocessed_df_path: Path,
    model_type: str,
    logger: MyLogger = DEF_SCRIPT_LOGGER,
    random_state: int = DEF_RANDOM_STATE,
    experiment_id: int = None,
    max_rows: int = None,
    n_workers: int = 1,
    # scale_pos_weight: float = 1.0,
):
    if logger == DEF_SCRIPT_LOGGER:
        # If default logger is used, start a new session
        logger.start_session()

    # logger.log_check("Starting hyperparameter tuning phase...")
    logger.log_check(f"Loading preprocessed data from: {preprocessed_df_path}")
    # logger.logger.info(
    #     f"Experiment ID: {expperiment_id}"
    #     if expperiment_id
    #     else "No Experiment ID provided."
    # )
    log_experiment_id(logger=logger, experiment_id=experiment_id)

    preprocessed_df_versioner = VersionedFileManager(
        file_path=preprocessed_df_path, logger=logger
    )
    preprocessed_df = load_df(
        df_file_path=preprocessed_df_versioner.current_newest, logger=logger
    )

    preprocessed_df = drop_cols(
        df=preprocessed_df, cols=TUNING_DROP_COLS, logger=logger
    )

    if max_rows:
        logger.log_check(f"Restricting dataset to first {max_rows} rows.")
        preprocessed_df = preprocessed_df.head(max_rows)

    X_train, y_train = (
        preprocessed_df.drop(columns=[TARGET]),
        preprocessed_df[TARGET],
    )

    model_wrapper, step_name = ModelWrapperFactory.create(
        model_type=model_type.lower(),
        random_state=random_state,
        logger=logger,
        scale_pos_weight=(
            XGBWrapper.calc_scale_pos_weight(y_train)
            if model_type.lower() == "xgb"
            else None
        ),
    )
    model = model_wrapper.get_model()

    tuning_wrapper = ModelTuningFactory.create(
        model_type=model_type,
        model=model,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        logger=logger,
        n_workers=n_workers,
    )

    best_params, best_score = tuning_wrapper.run_grid_search()
    clean_params = {
        key.replace("model__", ""): value 
        for key, value in best_params.items()
    }
    log_selected_features(tuning_wrapper.grid_search, logger=logger)
        
    logger.log_result(
        f"Tuning completed for model '{model_type}'. Best Score: {best_score}, Best Params (Cleaned): {clean_params}",
        print_to_console=True,
    )
    model.set_params(**clean_params)

    model_versioner = VersionedFileManager(
        file_path=TUNED_DIR / f"{step_name}_model_tuned.pkl", logger=logger
    )

    save_model(model=model, path=model_versioner.next_base_output, logger=logger)

    grid_search_dir =  MODEL_DIR / "grid_search"
    grid_search_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuning_wrapper.grid_search, grid_search_dir / "grid_search.pkl")


    return model_versioner.next_base_output


RANDOM_STATE = DEF_RANDOM_STATE

ARG_RESOLVERS = {
    "n_workers": lambda args: (
        (int)(get_n_jobs(reserve=2)) if args.workers < 0 else args.workers
    ),
    "max_rows": lambda args: args.max_rows,
    "random_state": lambda args: RANDOM_STATE,
    "model_type": lambda args: args.model,
}

def validate_workers(args: argparse.Namespace) -> None:
    if args.workers == 0:
        raise ValueError(
            f"Invalid workers argument: {args.workers}. Cannot equal zero."
        )
    
VALIDATORS = [
    validate_workers,
]    

# resolved_kwargs = {name: resolver(args) for name, resolver in ARG_RESOLVERS.items()}

def build_kwargs(args):
    return {name: resolver(args) for name, resolver in ARG_RESOLVERS.items()}


if __name__ == "__main__":
  
    import pandas as pd
    from src_code.ml_pipeline.training.tuning import SupportedModels, get_args
    from notebooks.logging_config import MyLogger

    parser = MyParser(description="Model Tuning")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_args(SupportedModels),
        help="Type of model to tune: 'RF' for Random Forest, 'XGB' for XGBoost",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        required=False,
        default=None,
        help="Limit dataset to first n rows only for testing purposes.",
    )

    parser.add_argument(
        "--reserve-cores",
        type=int,
        required=False,
        default=2,
        help="Reserve cores when using workers for parallel processing",
    )

    parser.add_argument(
        "--workers",
        type=int,
        required=False,
        default=1
        ,
        help="Use mutliple cores for parallel processing",
    )

    # script_logger = MyLogger(
    #     label="TUNING",
    #     section_name="TUNING LOGGER SCRIPT",
    #     file_log_path=LOG_DIR / "tuning_log.log",
    # )
    script_logger = DEF_SCRIPT_LOGGER
    script_logger.start_session(session_id=random.randint(1000, 9999))
    args = parser.parse_args()

    parser.validate(args=args, validators=VALIDATORS)
    # resolved_kwargs = build_kwargs(args)
    resolved_kwargs = parser.resolve_args(args=args, resolvers=ARG_RESOLVERS)

    # if args.workers == 0:
    #     raise ValueError(
    #         f"Invalid workers argument: {args.workers}. Cannot equal to zero."
    #     )

    model_type = args.model.lower()
    best_model_path = tune_hyperparams(
        preprocessed_df_path=PROCESSED_DATA_DIR / "train_engineered",
        # model_type=model_type,
        logger=script_logger,
        # random_state=RANDOM_STATE,
        # max_rows=args.max_rows,
        # n_workers=get_n_jobs(reserve=2) if args.workers < 0 else args.workers,
        **resolved_kwargs,
        # scale_pos_weight=scale_pos_weight,
    )
