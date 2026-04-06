


from notebooks.logging_config import MyLogger
from src_code.config import TUNED_DIR
from src_code.ml_pipeline.data_utils import load_artifact


logger = MyLogger(
    label="TRAINING_SCRIPT",
    section_name="TRAINING SCRIPT LOGGER",
    file_log_path=TUNED_DIR / "training_script_log.log",
)

models = ["RF", "XGB", "NN"]

for model_type in models:
    
    tuned_hyperparams = load_artifact(
        dir=TUNED_DIR,
        artifact_type="tuning-hyperparams",
        logger=logger,
        label=model_type,
    )

    logger.log_result(f"Loaded tuned hyperparameters for {model_type}: {tuned_hyperparams.hyperparams}")
