from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER


class RFPipelineWrapper:
    def __init__(self, rf: RandomForestClassifier, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
        # if logger:
        logger.log_check("Defining RF Pipeline...")
        self.pipeline = Pipeline(
            steps=[
                # ("preprocess", preprocessor),
                ("rf", rf)
            ]
        )

        # if logger:
        logger.log_result("Pipeline definition done.")

    def get_pipeline(self) -> Pipeline:
        return self.pipeline
