from notebooks.logging_config import NotebookLogger
from src_code.config import LOG_DIR

DEF_LOG_FILE = LOG_DIR / "logs_default.log"
DEF_NOTEBOOK_LOGGER = NotebookLogger(label="DEF LOGGER", notebook_name=DEF_LOG_FILE)

# DEF_RANDOM_STATE = 42 