from abc import abstractmethod
from collections import Counter
import time
import numpy as np
from pyparsing import ABC
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER, DEF_RANDOM_STATE
from src_code.ml_pipeline.preprocessing.transform import build_transformer
from src_code.utils.utils import timeit


DEF_N_JOBS = 1  # 🔴 IMPORTANT – consistent across models


class ModelWrapperBase(ABC):
    def __init__(
        self,
        random_state: int = DEF_RANDOM_STATE,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
    ):
        self.logger = logger
        self.model: BaseEstimator = None
        self.random_state = random_state

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    def set_up_pipeline(self):
        self.pipeline = Pipeline(
            [
                (
                    "prep",
                    build_transformer(
                        random_state=self.random_state, logger=self.logger
                    ),
                ),
                # ("cast_to_float", FunctionTransformer(to_numeric_df)), # Add this!
                (
                    "select",
                    SelectKBest(score_func=f_classif, k=100),
                ),  # Keep only top 100 features
                ("model", self.model),
            ],
            # memory=memory,
        )


class RFWrapper(ModelWrapperBase):

    def __init__(self, random_state: int, logger: MyLogger = DEF_NOTEBOOK_LOGGER):
        super().__init__(random_state, logger)

        self.logger.log_check("Defining Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            # min_samples_leaf=2,
            random_state=random_state,
            max_features="log2",
            min_samples_split=2,
            # class_weight=CLASS_WEIGHT,
            n_jobs=DEF_N_JOBS,  # 🔴 IMPORTANT
        )

        self.set_up_pipeline()

        # if logger:
        logger.log_result("Model definition done.")

    def get_model(self):
        return self.model

    def set_model(self, rf: RandomForestClassifier):
        self.model = rf

    @timeit(process_name="RF Fit")
    def fit(self, X_train, y_train):
        # self.logger.log_check("Starting RF fit...")
        # start = time.time()
        for col in X_train.columns:
            sample_val = X_train[col].iloc[0]
            if isinstance(sample_val, (list, np.ndarray, tuple)):
                print(f"Column '{col}' contains sequences, not scalars!")

        self.pipeline.fit(X_train, y_train)
        # self.model.fit(X_train, y_train)
        # end = time.time()
        # self.logger.log_result(f"RF fit completed. Time: {end - start:2f}")


class XGBWrapper(ModelWrapperBase):

    @staticmethod
    def calc_scale_pos_weight(y):
        """
        Calculate scale_pos_weight for XGBoost based on class distribution.
        Args:
            y (series): _target labels

        Returns:
            float: _scale_pos_weight value
        """
        counter = Counter(y)
        scale_pos_weight = counter[0] / counter[1]  # weight = #negatives / #positives
        return scale_pos_weight

    DEF_N_ESTIMATORS = 300
    DEF_LEARNING_RATE = 0.1

    ES_N_ESTIMATORS = 3000
    ES_LEARNING_RATE = 0.05

    def __init__(
        self,
        random_state: int,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
        scale_pos_weight=None,
    ):
        super().__init__(random_state, logger)

        self.logger.log_check("Defining XGBoost...")

        self.model = XGBClassifier(
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
            # early_stopping_rounds=20,
            scale_pos_weight=scale_pos_weight,
        )
        self.set_up_pipeline()

        self.logger.log_result("XGBoost definition done.")

    def get_model(self):
        return self.model

    def set_model(self, model: XGBClassifier):
        self.model = model

    @timeit(process_name="XGB Fit")
    def fit(self, X_train, y_train, X_val, y_val):
        # self.logger.log_check("Starting XGBoost fit...")
        # start = time.time()

        # 1. We need to manually transform the validation set
        # so it matches what the model expects after 'prep' and 'select'
        # We use the pipeline's internal steps to do this
        X_val_transformed = self.pipeline[:-1].fit_transform(
            X_train, y_train
        )  # Fit on train
        X_val_ready = self.pipeline[:-1].transform(X_val)  # Transform validation

        # self.pipeline.fit(
        #     X_train,
        #     y_train,
        #     eval_set=[(X_val, y_val)],
        #     # early_stopping_rounds=20,
        #     verbose=False,
        # )
        # 2. Pass the parameters with the model__ prefix
        self.pipeline.fit(
            X_train,
            y_train,
            # model__eval_set=[(X_val_ready, y_val)],
            model__eval_set=[(X_val_ready, y_val)],
            model__verbose=False,
        )
        # end = time.time()
        # self.logger.log_result(f"XGBoost fit completed. Time: {end - start:2f}")


class ModelWrapperFactory:
    @staticmethod
    def create(
        model_type: str,
        random_state: int,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
        scale_pos_weight=None,
    ):
        if model_type.lower() == "rf":
            return RFWrapper(random_state, logger=logger), "rf"
        if model_type.lower() == "xgb":
            return (
                XGBWrapper(
                    random_state, logger=logger, scale_pos_weight=scale_pos_weight
                ),
                "xgb",
            )
