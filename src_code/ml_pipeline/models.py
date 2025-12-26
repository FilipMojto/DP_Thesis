from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER


# rf = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=20,
#     # min_samples_leaf=2,
#     random_state=RANDOM_STATE,
#     max_features="log2",
#     min_samples_split=2,
#     # class_weight=CLASS_WEIGHT,
#     n_jobs=1,  # 🔴 IMPORTANT
# )


class RFWrapper:
    def __init__(self, random_state: int, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
        super().__init__()

        # if logger:
        logger.log_check("Defining Random Forest...")

        self.rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            # min_samples_leaf=2,
            random_state=random_state,
            max_features="log2",
            min_samples_split=2,
            # class_weight=CLASS_WEIGHT,
            n_jobs=1,  # 🔴 IMPORTANT
        )

        # if logger:
        logger.log_result("Model definition done.")

    def get_model(self):
        return self.rf


