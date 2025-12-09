import logging
from logging.handlers import RotatingFileHandler
import os
import random

import ipynbname

LOG_PATH = os.path.join(os.getcwd(), "logs/notebooks.log")

# def create_logger(name: str = "nb"):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)

#     # Avoid adding duplicate handlers when re-imported in Jupyter
#     if not logger.handlers:
#         handler = RotatingFileHandler(
#             LOG_PATH,
#             maxBytes=5_000_000,  # 5MB log rotation
#             backupCount=3,
#             encoding="utf-8"
#         )
#         formatter = logging.Formatter(
#             "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#         )
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)

#     return logger


# def set_up_logger(logger):
#     # Get notebook name
#     try:
#         notebook_name = ipynbname.name()
#     except:
#         notebook_name = "UNKNOWN_NOTEBOOK"

#     # Generate small session run ID
#     session_id = random.randint(100, 999)

#     # Create logger with session ID included
#     logger_name = f"{notebook_name}-S{session_id}"
#     logger = create_logger(logger_name)

#     def log_check(text: str):
#         logger.info(f"[EDA CHECK] {text}")

#     def log_result(text: str):
#         logger.info(f"[EDA RESULT] {text}")

#     return log_check, log_result

# Combined setup function
def setup_notebook_logging(log_path: str = LOG_PATH, label: str = None):
    # 1. Get name and create logger instance
    try:
        notebook_name = ipynbname.name()
    except:
        notebook_name = "UNKNOWN_NOTEBOOK"
    
    session_id = random.randint(100, 999)
    logger_name = f"{notebook_name}-S{session_id}"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 2. Add handler logic (using the passed log_path or default)
    if not logger.handlers:
        handler = RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8"
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # 3. Define and return closures
    def log_check(text: str, print_to_console: bool = False):
        msg = f"[{label if label else "UNSPECIFIED"} CHECK] {text}"
        logger.info(msg)

        if print_to_console:
            print(msg)


    def log_result(text: str, print_to_console: bool = False):
        msg = f"[{label if label else "UNSPECIFIED"} RESULT] {text}"
        logger.info(msg)

        if print_to_console:
            print(msg)

    def log_start(print_to_console: bool = False):
        msg = f"================== Starting notebook: {notebook_name} (Session {session_id}) =================="
        logger.info(msg)

        if print_to_console:
            print(msg)


    return logger, log_start, log_check, log_result