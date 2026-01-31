
from notebooks.logging_config import MyLogger


def log_experiment_id(logger: MyLogger, experiment_id: int):
    logger.log_check(
        f"Experiment ID: {experiment_id}"
        if experiment_id
        else "No Experiment ID provided."
    )