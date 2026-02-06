import os

import joblib

from notebooks.logging_config import MyLogger
from src_code.config import MODEL_DIR
from src_code.ml_pipeline.training.tuning import log_selected_features


grid_search = joblib.load(MODEL_DIR / "grid_search" / "grid_search.pkl")

logger = MyLogger(label="Tets", section_name="Tets", file_log_path="test.log")

log_selected_features(grid_search=grid_search, logger=logger, top_n=100)