from pathlib import Path
from notebooks.constants import TARGET
from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERING_MAPPINGS, MODEL_DIR
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER, DEF_RANDOM_STATE
from src_code.ml_pipeline.data_utils import load_df, save_model
from src_code.ml_pipeline.models import ModelWrapperFactory, XGBWrapper

from src_code.ml_pipeline.training.tuning import ModelTuningFactory
from src_code.versioning import VersionedFileManager

DEF_SCRIPT_LOGGER = MyLogger(
    label="TUNING",
    section_name="TUNING LOGGER SCRIPT",
    file_log_path=LOG_DIR / "tuning_log.log",
)


def tune_hyperparams(
    preprocessed_df_path: Path,
    model_type: str,
    logger: MyLogger = DEF_SCRIPT_LOGGER,
    random_state: int = DEF_RANDOM_STATE,
    expperiment_id: int = None,
    # scale_pos_weight: float = 1.0,
):
    if logger == DEF_SCRIPT_LOGGER:
        # If default logger is used, start a new session
        logger.start_session()

    logger.log_check("Starting hyperparameter tuning phase...")
    logger.log_check(f"Loading preprocessed data from: {preprocessed_df_path}")
    logger.logger.info(
        f"Experiment ID: {expperiment_id}"
        if expperiment_id
        else "No Experiment ID provided."
    )

    preprocessed_df_versioner = VersionedFileManager(
        file_path=preprocessed_df_path, logger=logger
    )
    preprocessed_df = load_df(
        df_file_path=preprocessed_df_versioner.current_newest, logger=logger
    )

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
    )

    best_params, best_score = tuning_wrapper.run_grid_search()
    logger.log_result(
        f"Tuning completed for model '{model_type}'. Best Score: {best_score}, Best Params: {best_params}",
        print_to_console=True,
    )
    model.set_params(**best_params)

    model_versioner = VersionedFileManager(
        file_path=MODEL_DIR / f"{step_name}_model_tuned.pkl", logger=logger
    )

    save_model(model=model, model_file_path=model_versioner.next_base_output, logger=logger)

    return model_versioner.next_base_output


RANDOM_STATE = DEF_RANDOM_STATE

if __name__ == "__main__":
    import argparse
    import random
    from src_code.config import LOG_DIR
    import pandas as pd
    from src_code.ml_pipeline.training.tuning import SupportedModels, get_args
    from notebooks.logging_config import MyLogger

    args = argparse.ArgumentParser(description="Model Tuning")
    args.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_args(SupportedModels),
        help="Type of model to tune: 'rf' for Random Forest, 'xgb' for XGBoost",
    )
    script_logger = MyLogger(
        label="TUNING",
        section_name="TUNING LOGGER SCRIPT",
        file_log_path=LOG_DIR / "tuning_log.log",
    )
    script_logger.start_session(session_id=random.randint(1000, 9999))
    parsed_args = args.parse_args()

    # dataset: pd.DataFrame = load_df(
    #     df_file_path=ENGINEERING_MAPPINGS["train"]["output"]
    # )

    model_type = parsed_args.model.lower()
    best_params, best_score = tune_hyperparams(
        preprocessed_df_path=ENGINEERING_MAPPINGS["train"]["output"],
        model_type=model_type,
        logger=script_logger,
        random_state=RANDOM_STATE,
        # scale_pos_weight=scale_pos_weight,
    )
