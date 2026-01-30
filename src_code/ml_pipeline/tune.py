from pathlib import Path
from notebooks.constants import TARGET
from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERING_MAPPINGS
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER, DEF_RANDOM_STATE
from src_code.ml_pipeline.data_utils import load_df
from src_code.ml_pipeline.models import ModelWrapperFactory, XGBWrapper

from src_code.ml_pipeline.training.tuning import ModelTuningFactory
from src_code.versioning import VersionedFileManager


def main(
    preprocessed_df_path: Path,
    model_type: str,
    logger: MyLogger = DEF_NOTEBOOK_LOGGER,
    random_state: int = 42,
    # scale_pos_weight: float = 1.0,
):
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
        scale_pos_weight=XGBWrapper.calc_scale_pos_weight(y_train) if model_type.lower() == "xgb" else None,
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

    return best_params, best_score


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
    best_params, best_score = main(
        preprocessed_df_path=ENGINEERING_MAPPINGS["train"]["output"],
        model_type=model_type,
        logger=script_logger,
        random_state=RANDOM_STATE,
        # scale_pos_weight=scale_pos_weight,
    )
