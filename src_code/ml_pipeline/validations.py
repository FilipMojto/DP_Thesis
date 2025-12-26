import time
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.utils import get_n_jobs


class CVWrapper:
    N_SPLITS = 10

    def __init__(
        self,
        random_state: int,
        logger: NotebookLogger = DEF_NOTEBOOK_LOGGER,
        reserve_cores: int = 2,
    ):
        # -------------------------------------------------------------------------
        # 1. Define the KFold splitter (non-stratified)
        # -------------------------------------------------------------------------
        # NOTE: This does NOT guarantee equal class proportions in each fold.
        self.logger = logger
        self.logger.log_check("Defining cross-validation...")

        self.cv = KFold(n_splits=CVWrapper.N_SPLITS, shuffle=True, random_state=random_state)

        # -------------------------------------------------------------------------
        # 2. Define the scoring metrics
        # -------------------------------------------------------------------------
        self.scoring = {
            "roc_auc": "roc_auc",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
        }

        self.n_jobs = min(CVWrapper.N_SPLITS, get_n_jobs(reserve_cores))
        self.logger.log_result("Validation definition done.")

    def cross_validate(self, model, X_train, y_train):
        self.logger.log_check("Cross validating provided model...")
        start = time.time()

        # -------------------------------------------------------------------------
        # 3. Execute the cross-validation
        # -------------------------------------------------------------------------
        self.cv_results = cross_validate(
            model,  # Your machine learning pipeline
            X_train,  # Training features
            y_train,  # Training labels
            cv=self.cv,  # The KFold splitter
            scoring=self.scoring,
            # n_jobs=6,  # Use CPU cores
            n_jobs=self.n_jobs,
            verbose=3,
        )

        end = time.time()

        self.logger.log_result(f"Validation completed. Time: {end-start:2f}")
        
        return self.cv_results

    def mean_results(self):
        # -------------------------------------------------------------------------
        # 4. View and aggregate the results
        # -------------------------------------------------------------------------
        # The mean of the results gives the model's average performance.
        average_metrics = pd.DataFrame(self.cv_results).mean()

        self.logger.log_result("\n--- Average Cross-Validation Metrics (KFold) ---")
        self.logger.log_result(average_metrics)
