from abc import abstractmethod
import time
from pyparsing import ABC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from notebooks.logging_config import MyLogger
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

DEF_N_JOBS = 1  # 🔴 IMPORTANT – consistent across models

class ModelWrapperBase(ABC):
    def __init__(self, random_state: int, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
        self.logger = logger

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

class RFWrapper(ModelWrapperBase):
    def __init__(self, random_state: int, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
        super().__init__(random_state, logger)

        # if logger:
        # self.logger = logger
        self.logger.log_check("Defining Random Forest...")

        self.rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            # min_samples_leaf=2,
            random_state=random_state,
            max_features="log2",
            min_samples_split=2,
            # class_weight=CLASS_WEIGHT,
            n_jobs=DEF_N_JOBS,  # 🔴 IMPORTANT
        )

        # if logger:
        logger.log_result("Model definition done.")

    def get_model(self):
        return self.rf

    def fit(self, X_train, y_train):
        self.logger.log_check("Starting RF fit...")
        start = time()
        self.rf.fit(X_train, y_train)
        end = time()
        self.logger.log_result(f"RF fit completed. Time: {end - start:2f}")


class XGBWrapper(ModelWrapperBase):
    # DEF_N_ESTIMATORS = 300
    # DEF_LEARNING_RATE = 0.1

    # # here ES means early stopping
    # ES_N_ESTIMATORS = 3000
    # ES_LEARNING_RATE = 0.05

    # def __init__(self, random_state: int, logger: NotebookLogger = DEF_NOTEBOOK_LOGGER):
    #     self.logger = logger
    #     self.logger.log_check("Defining XGBoost...")

    #     self.xgb = XGBClassifier(
    #         n_estimators=self.DEF_N_ESTIMATORS,
    #         max_depth=6,
    #         learning_rate=self.DEF_LEARNING_RATE,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         objective="binary:logistic",  # change if multiclass
    #         eval_metric="logloss",
    #         random_state=random_state,
    #         n_jobs=DEF_N_JOBS,  # 🔴 IMPORTANT – consistent with RF
    #         tree_method="hist",  # fast & stable on CPU
    #     )

    #     self.logger.log_result("XGBoost definition done.")

    # def get_model(self):
    #     return self.xgb
    DEF_N_ESTIMATORS = 300
    DEF_LEARNING_RATE = 0.1

    ES_N_ESTIMATORS = 3000
    ES_LEARNING_RATE = 0.05

    def __init__(self, random_state: int, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
        super().__init__(random_state, logger)

        # self.logger = logger
        self.logger.log_check("Defining XGBoost...")

        self.xgb = XGBClassifier(
            n_estimators=self.DEF_N_ESTIMATORS,
            max_depth=6,
            learning_rate=self.DEF_LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=DEF_N_JOBS,
            tree_method="hist",
            early_stopping_rounds=20,
        )

        self.logger.log_result("XGBoost definition done.")

    def get_model(self):
        return self.xgb

    def fit(self, X_train, y_train, X_val, y_val):
        self.logger.log_check("Starting XGBoost fit...")
        start = time.time()
        self.xgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            # early_stopping_rounds=20,
            verbose=3,
        )
        end = time.time()
        self.logger.log_result(f"XGBoost fit completed. Time: {end - start:2f}")

    # def enable_early_stopping(self, X_val, y_val):
    #     self.xgb.set_params(
    #         n_estimators=self.ES_N_ESTIMATORS,
    #         learning_rate=self.ES_LEARNING_RATE,
    #     )
    #     self.X_val = X_val
    #     self.y_val = y_val
    #     self.logger.log_result("Early stopping enabled for XGBoost.")

    # def disable_early_stopping(self):
    #     self.xgb.set_params(
    #         n_estimators=self.DEF_N_ESTIMATORS,
    #         learning_rate=self.DEF_LEARNING_RATE,
    #     )
    #     self.X_val = None
    #     self.y_val = None
    #     self.logger.log_result("Early stopping disabled for XGBoost.")


class ModelWrapperFactory:
    @staticmethod
    def create(model_type: str, random_state: int):
        if model_type == "rf":
            return RFWrapper(random_state), "rf"
        if model_type == "xgb":
            return XGBWrapper(random_state), "xgb"
