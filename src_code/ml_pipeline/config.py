from notebooks.logging_config import MyLogger
from src_code.config import LOG_DIR

DEF_LOG_FILE = LOG_DIR / "logs_default.log"
DEF_NOTEBOOK_LOGGER = MyLogger(label="DEF LOGGER", section_name=DEF_LOG_FILE)

# DEF_RANDOM_STATE = 42 