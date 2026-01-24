import random
import time
from typing import get_args
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from abc import ABC, abstractmethod


from notebooks.constants import TARGET
from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERING_MAPPINGS, LOG_DIR, SupportedModels
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.data_utils import load_df
from src_code.ml_pipeline.preprocessing.feature_config import DROP_COLS
from src_code.ml_pipeline.models import ModelWrapperFactory

# from src_code.ml_pipeline.training.utils import drop_cols
from src_code.ml_pipeline.utils import get_n_jobs
from src_code.utils.utils import timeit


class TuningWrapperBase(ABC):
    def __init__(self, X_train, y_train, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
        self.X_train = X_train
        self.y_train = y_train
        self.logger = logger

    @abstractmethod
    def run_grid_search(self):
        pass

    def print_results(self, best_params: dict, best_score: float):
        # self.logger.log_result(f"Search complete. Time: {end - start:2f}")
        self.logger.log_result(f"Best Parameters: {best_params}")
        self.logger.log_result(f"Best MCC Score: {best_score}")


class RFTuningWrapper(TuningWrapperBase):
    def __init__(
        self,
        rf: RandomForestClassifier,
        X_train,
        y_train,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
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

    @timeit(process_name="RF - Grid Search")
    def run_grid_search(self):
        # 5. Run the search
        self.logger.log_check("Running grid search...")

        # start = time.time()
        self.grid_search.fit(self.X_train, self.y_train)
        # end = time.time()

        best_params = self.grid_search.best_params_
        best_score = self.grid_search.best_score_

        # self.logger.log_result(f"Search complete. Time: {end - start:2f}")
        # self.logger.log_result(f"Best Parameters: {best_params}")
        # self.logger.log_result(f"Best MCC Score: {best_score}")
        self.print_results(best_params=best_params, best_score=best_score)

        return best_params, best_score


class XGBTuningWrapper(TuningWrapperBase):
    def __init__(
        self,
        xgb: XGBClassifier,
        X_train,
        y_train,
        X_val,
        y_val,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
    ):
        self.xgb = xgb
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.logger = logger

    @timeit(process_name="XGB - Grid Search")
    def run_grid_search(self):
        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        self.grid_search = GridSearchCV(
            estimator=self.xgb,
            param_grid=param_grid,
            scoring="roc_auc",  # or MCC if you use it
            cv=3,
            n_jobs=1,
            verbose=3,
        )

        # grid.fit(self.X_train, self.y_train)
        self.grid_search.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )

        
        best_params = self.grid_search.best_params_
        best_score = self.grid_search.best_score_

        self.print_results(best_params=best_params, best_score=best_score)

        return best_params, best_score


class ModelTuningFactory:
    @staticmethod
    def create(model_type: str, model, X_train, y_train, X_val=None, y_val=None, logger = DEF_NOTEBOOK_LOGGER):
        if model_type == "rf":
            return RFTuningWrapper(rf=model, X_train=X_train, y_train=y_train, logger=logger)
        if model_type == "xgb":
            return XGBTuningWrapper(
                xgb=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, logger=logger


import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Model Tuning")
    args.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_args(SupportedModels),
        help="Type of model to tune: 'rf' for Random Forest, 'xgb' for XGBoost",
    )
    script_logger = MyLogger(
        label="TUNING",
        section_name="TUNING LOGGER SCRIPT",
        file_log_path=LOG_DIR / "tuning_log.log",
    )
    script_logger.start_session(session_id=random.randint(1000, 9999))
    parsed_args = args.parse_args()

    dataset: pd.DataFrame = load_df(
        df_file_path=ENGINEERING_MAPPINGS["train"]["output"]
    )
    X_train = dataset.drop(columns=[TARGET])
    y_train = dataset[TARGET]

    # X_train = drop_cols(df=X_train, cols=DROP_COLS, logger=script_logger)

    model_type = parsed_args.model.lower()

    model_wrapper = ModelWrapperFactory.create(model_type=model_type, random_state=42)[
        0
    ]
    model = model_wrapper.get_model()

    tuner = ModelTuningFactory.create(
        model_type=model_type, model=model, X_train=X_train, y_train=y_train
    )

    best_params, best_score = tuner.run_grid_search()
    script_logger.log_result(
        f"Tuning completed for {model_type.upper()}. Best Params: {best_params}, Best Score: {best_score}"
    )
