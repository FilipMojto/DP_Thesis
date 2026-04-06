from pathlib import Path
import argparse
import random
from typing import Any, Callable, Dict, get_args

import joblib
import pandas as pd

from src_code.ml_pipeline.resources import (
    DEF_CORE_MODE_TYPE,
    DEF_NUM_OF_CORES,
    DEF_RESERVE_CORES,
    CoreConfig,
    CoreModeType,
)
from src_code.ml_pipeline.training.constants import DEF_TOP_K
from src_code.ml_pipeline.training.tuning import SupportedModel
from notebooks.logging_config import MyLogger
from notebooks.constants import TARGET
from notebooks.logging_config import MyLogger
from src_code.config import (
    ENGINEERED_DATA_DIR,
    # ENGINEERING_MAPPINGS,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    TUNED_DIR,
)
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER, DEF_RANDOM_STATE
from src_code.ml_pipeline.data_utils import (
    PipelineArtifact,
    load_df,
    save_artifact,
    save_model,
)
from src_code.ml_pipeline.experimenting.types import (
    ARG_RESOLVERS_COLL,
    ARG_VALIDATOR,
    ARG_VALIDATORS_COLL,
    TuningResults,
)
from src_code.ml_pipeline.experimenting.utils import MyParser, log_experiment_id
from src_code.ml_pipeline.models import ModelWrapperFactory, XGBWrapper
from src_code.config import LOG_DIR
from src_code.ml_pipeline.preprocessing.feature_config import (
    DROP_COLS,
    TUNING_DROP_COLS,
)
from src_code.ml_pipeline.preprocessing.preprocessing import drop_cols
from src_code.ml_pipeline.training.tuning import (
    ModelTuningFactory,
    log_selected_features,
)

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
    model_type: SupportedModel,
    preprocessed_df_path: Path = Path(ENGINEERED_DATA_DIR / "train_engineered.feather"),
    logger: MyLogger = DEF_SCRIPT_LOGGER,
    random_state: int = DEF_RANDOM_STATE,
    experiment_id: int = None,
    max_rows: int = None,
    core_mode: CoreModeType = DEF_CORE_MODE_TYPE,
    n_cores: int = DEF_NUM_OF_CORES,
    reserve_cores: int = DEF_RESERVE_CORES,
    top_k: int = DEF_TOP_K,
    # scale_pos_weight: float = 1.0,
):
    logger.start_session(
        session_id=experiment_id if experiment_id else MyLogger.DEF_SESSION_ID
    )
    # if logger == DEF_SCRIPT_LOGGER:
    #     # If default logger is used, start a new session
    #     logger.start_session()

    log_experiment_id(logger=logger, experiment_id=experiment_id)
    # logger.log_result("")
    logger.log_result(
        f"Config: [{model_type=}, {random_state=}, {experiment_id=}, {max_rows=}, {n_cores=}]"
    )

    results = TuningResults()

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

    model_wrapper = ModelWrapperFactory.create(
        model_type=model_type,
        random_state=random_state,
        logger=logger,
        scale_pos_weight=(
            XGBWrapper.calc_scale_pos_weight(y_train) if model_type == "XGB" else None
        ),
        top_k=top_k,
    )
    model = model_wrapper.get_model()

    tuning_wrapper = ModelTuningFactory.create(
        model_type=model_type,
        model=model,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        logger=logger,
        core_config=CoreConfig(
            reserve_cores=reserve_cores, num_of_cores=n_cores, mode=core_mode
        ),
    )

    tuning_wrapper.run_grid_search()
    best_params, best_score = tuning_wrapper.get_best_score()

    clean_params = {
        key.replace("model__", ""): value for key, value in best_params.items()
    }
    log_selected_features(tuning_wrapper.grid_search, logger=logger)

    logger.log_result(
        f"Tuning completed for model '{model_type}'. Best Score: {best_score}, Best Params (Cleaned): {clean_params}",
        print_to_console=True,
    )
    # model.set_params(**clean_params)

    # model_versioner = VersionedFileManager(
    #     file_path=TUNED_DIR / f"{model_type}_model_tuned.pkl", logger=logger
    # )

    best_fitted_model = tuning_wrapper.grid_search.best_estimator_
    trained_features = best_fitted_model.feature_names_in_
    logger.log_result(f"The model was trained on features: {trained_features}")
    results.features_trained_on = len(trained_features)

    # save_model(model=model, path=model_versioner.next_base_output, logger=logger)
    # save_model(
    #     model=best_params, path=model_versioner.next_base_output, logger=logger
    # )
    results.param_artifact = save_artifact(
        dir=TUNED_DIR,
        artifact=PipelineArtifact(
            label=f"{model_type}",
            artifact_type="tuning-hyperparams",
            hyperparams=best_params,
        ),
        logger=logger,
    )
    # results.param_artifact = artifact_path

    grid_search_dir = MODEL_DIR / "grid_search"
    grid_search_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuning_wrapper.grid_search, grid_search_dir / "grid_search.pkl")

    return results


RANDOM_STATE = DEF_RANDOM_STATE

# ARG_RESOLVERS: Dict[str, Callable[[argparse.Namespace], Any]] = {
#     "n_workers": lambda args: (
#         (int)(get_n_jobs(reserve=2)) if args.workers < 0 else args.workers
#     ),
#     "max_rows": lambda args: args.max_rows,
#     "random_state": lambda args: RANDOM_STATE,
#     "model_type": lambda args: args.model,
# }
ARG_RESOLVERS: ARG_RESOLVERS_COLL = {
    "n_workers": lambda cfg: (
        int(get_n_jobs(reserve=2)) if cfg["workers"] < 0 else cfg["workers"]
    ),
    "max_rows": lambda cfg: cfg["max_rows"],
    "random_state": lambda cfg: RANDOM_STATE,
    "model_type": lambda cfg: cfg["model"],
}


# def validate_workers(args: argparse.Namespace) -> None:
#     if args.workers == 0:
#         raise ValueError(
#             f"Invalid workers argument: {args.workers}. Cannot equal zero."
#         )
def validate_workers(cfg: Dict[str, Any]) -> None:
    if cfg.get("workers", 1) == 0:
        raise ValueError("workers cannot be zero")


ARG_VALIDATORS: ARG_VALIDATORS_COLL = [
    validate_workers,
]

# resolved_kwargs = {name: resolver(args) for name, resolver in ARG_RESOLVERS.items()}


def build_kwargs(args):
    return {name: resolver(args) for name, resolver in ARG_RESOLVERS.items()}


def get_parser(add_help: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help, description="Model Tuning")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_args(SupportedModel),
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
        default=DEF_RESERVE_CORES,
        help=f"Reserve cores when using workers for parallel processing. Defaults to {DEF_RESERVE_CORES}.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        required=False,
        default=DEF_NUM_OF_CORES,
        help=f"Use multiple cores for parallel processing. Defaults to {DEF_NUM_OF_CORES}.",
    )

    parser.add_argument(
        "--core-mode",
        type=str,
        required=False,
        default=DEF_CORE_MODE_TYPE,
        choices=get_args(CoreModeType),
        help=f"""Use 'manual' for manual specification of cores used in the process, use 'all' for utilizing the maximum amount of cores.
          Defaults to '{DEF_CORE_MODE_TYPE}'.""",
    )

    return parser


if __name__ == "__main__":

    # parser = MyParser(description="Model Tuning")
    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     required=True,
    #     choices=get_args(SupportedModels),
    #     help="Type of model to tune: 'RF' for Random Forest, 'XGB' for XGBoost",
    # )

    # parser.add_argument(
    #     "--max-rows",
    #     type=int,
    #     required=False,
    #     default=None,
    #     help="Limit dataset to first n rows only for testing purposes.",
    # )

    # parser.add_argument(
    #     "--reserve-cores",
    #     type=int,
    #     required=False,
    #     default=2,
    #     help="Reserve cores when using workers for parallel processing",
    # )

    # parser.add_argument(
    #     "--workers",
    #     type=int,
    #     required=False,
    #     default=1,
    #     help="Use mutliple cores for parallel processing",
    # )

    # script_logger = MyLogger(
    #     label="TUNING",
    #     section_name="TUNING LOGGER SCRIPT",
    #     file_log_path=LOG_DIR / "tuning_log.log",
    # )
    parser = get_parser(add_help=True)

    logger = DEF_SCRIPT_LOGGER
    # logger.start_session(session_id=random.randint(1000, 9999))
    args = parser.parse_args()

    # parser.validate(args=args, validators=ARG_VALIDATORS)
    # resolved_kwargs = build_kwargs(args)
    # resolved_kwargs = parser.resolve_args(args=args, resolvers=ARG_RESOLVERS)

    # if args.workers == 0:
    #     raise ValueError(
    #         f"Invalid workers argument: {args.workers}. Cannot equal to zero."
    #     )

    model_type = args.model.lower()
    best_model_path = tune_hyperparams(
        preprocessed_df_path=PROCESSED_DATA_DIR / "train_engineered",
        # model_type=model_type,
        logger=logger,
        experiment_id=None,
        max_rows=args.max_rows,
        model_type=args.model,
        n_cores=args.workers,
        core_mode=args.core_mode,
        # # random_state=RANDOM_STATE,
        # # max_rows=args.max_rows,
        # # n_workers=get_n_jobs(reserve=2) if args.workers < 0 else args.workers,
        # **resolved_kwargs,
        # # scale_pos_weight=scale_pos_weight,
    )
