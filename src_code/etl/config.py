
from notebooks.logging_config import MyLogger
from src_code.config import LOG_DIR


DEF_ETL_LOGGER = MyLogger(label="ETL", section_name="Data Extraction", file_log_path=LOG_DIR / "etl_process.log")