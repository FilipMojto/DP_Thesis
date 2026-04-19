from typing import Dict

from typing_extensions import Literal, get_args

from notebooks.logging_config import MyLogger
from src_code.config import MODEL_DIR
import src_code.ml_pipeline.data_utils as dutls
from src_code.ml_pipeline.models import ModelWrapperBase


ENSEMBLE_SUPPORTED_MODELS = Literal['RF', 'XGB', 'NN']


def load_ensemble_supported_models(logger: MyLogger) -> Dict[str, ModelWrapperBase]:
    loaded_models = {
        model_type: dutls.load_artifact(
            dir=MODEL_DIR,
            artifact_type="trained_model",
            logger=logger,
            label=model_type,
        ).model_wrapper
        for model_type in get_args(ENSEMBLE_SUPPORTED_MODELS)
    }

    logger.log_result(f"Loaded base models for ensemble: {list(loaded_models.keys())}")
    return loaded_models