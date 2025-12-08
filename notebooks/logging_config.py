import logging
from logging.handlers import RotatingFileHandler
import os

LOG_PATH = os.path.join(os.getcwd(), "logs/notebooks.log")

def setup_logger(name: str = "nb"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers when re-imported in Jupyter
    if not logger.handlers:
        handler = RotatingFileHandler(
            LOG_PATH,
            maxBytes=5_000_000,  # 5MB log rotation
            backupCount=3,
            encoding="utf-8"
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger