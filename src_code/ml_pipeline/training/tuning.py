import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV

from notebooks.logging_config import NotebookLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.utils import get_n_jobs


class RFTuningWrapper:
    def __init__(
        self,
        rf: RandomForestClassifier,
        X_train,
        y_train,
        logger: NotebookLogger = DEF_NOTEBOOK_LOGGER,
        reserve_cores: int = 2,
    ):
        # 1. Define the model
        # rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        logger.log_check("Initializing RFTunning Wrapper object...")
        self.rf = rf
        self.X_train = X_train
        self.y_train = y_train
        self.logger = logger

        # 2. Define the parameter grid
        self.param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        }

        # 3. Create a custom scorer for MCC
        self.mcc_scorer = make_scorer(matthews_corrcoef)

        # 4. Set up Grid Search
        self.grid_search = GridSearchCV(
            estimator=rf,
            param_grid=self.param_grid,
            scoring=self.mcc_scorer,
            cv=5,  # 5-fold cross-validation
            n_jobs=get_n_jobs(reserve=reserve_cores),  # Use all CPU cores
            verbose=3,
        )

        self.logger.log_result("Initialization completed.")

    def run_grid_search(self):
        # 5. Run the search
        self.logger.log_check("Running grid search...")

        start = time.time()
        self.grid_search.fit(self.X_train, self.y_train)
        end = time.time()

        best_params = self.grid_search.best_params_
        best_score = self.grid_search.best_score_

        self.logger.log_result(f"Search complete. Time: {end - start:2f}")
        self.logger.log_result(f"Best Parameters: {best_params}")
        self.logger.log_result(f"Best MCC Score: {best_score}")

        return best_params, best_score
