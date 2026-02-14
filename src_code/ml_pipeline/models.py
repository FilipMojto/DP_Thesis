from abc import abstractmethod
from collections import Counter
import numpy as np
from pyparsing import ABC
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

import torch
from xgboost import XGBClassifier
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from notebooks.logging_config import MyLogger
from src_code.config import SupportedModel
from src_code.ml_pipeline.builders import PipelineBuilder
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER, DEF_RANDOM_STATE
from src_code.utils.utils import timeit


DEF_N_JOBS = 1  # 🔴 IMPORTANT – consistent across models


class ModelWrapperBase(ABC):
    def __init__(
        self,
        top_k: int,
        random_state: int = DEF_RANDOM_STATE,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
    ):
        self.logger = logger
        self.model: BaseEstimator = None
        self.random_state = random_state

        # 🔵 default behavior
        self.use_scaling = False
        self.top_k = top_k
        self.hyperparams: dict = None

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        pass

    # @abstractmethod
    # def configure(self, hyperparams: dict):
    #     pass

    # @abstractmethod
    # def predict(self, X):
    #     pass
    def transform(self, X):
        # def predict(self, X):
        return self.pipeline[:-1].transform(X).astype(np.float32)
        # return self.model.predict(X_trans)

    def set_up_pipeline(self):
        self.pipeline = PipelineBuilder.build(
            model=self.model,
            logger=self.logger,
            random_state=self.random_state,
            top_k=self.top_k,
            use_scaling=self.use_scaling,
        )


class RFWrapper(ModelWrapperBase):

    def __init__(
        self,
        random_state: int,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
        top_k: int = 100,
        tuned_params=None,
    ):
        super().__init__(random_state=random_state, logger=logger, top_k=top_k)

        self.logger.log_check("Defining Random Forest...")
        self.use_scaling = False
        # self.top_k = top_k

        if tuned_params:
            self.model = RandomForestClassifier(**tuned_params)
        else:
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

    # @timeit(process_name="RF Fit")
    # def fit(self, X_train, y_train):
    #     # self.logger.log_check("Starting RF fit...")
    #     # start = time.time()
    #     # for col in X_train.columns:
    #     #     sample_val = X_train[col].iloc[0]
    #     #     if isinstance(sample_val, (list, np.ndarray, tuple)):
    #     #         self.logger.(f"Column '{col}' contains sequences, not scalars!")

    #     self.pipeline.fit(X_train, y_train)
    #     self.feature_names_ = list(X_train.columns)
    @timeit(process_name="RF Fit")
    def fit(self, X_train, y_train):
        # 1️⃣ Fit the entire pipeline
        # (RF doesn't need an eval_set, so we can do this in one shot)
        # self.pipeline.fit(X_train, y_train)
        X_train_ready = self.pipeline[:-1].fit_transform(X_train, y_train)

        self.model.fit(X_train_ready, y_train)
        self.pipeline.steps[-1] = ("model", self.model)

        # 2️⃣ Sync the internal model reference
        # This ensures self.model is the fitted version from the pipeline
        # self.model = self.pipeline.named_steps["model"]

        # 3️⃣ Save the CORRECT processed feature names
        try:
            # This pulls names from the pipeline steps (encoding, selection, etc.)
            self.feature_names_ = self.pipeline[:-1].get_feature_names_out().tolist()
        except Exception as e:
            # Fallback if preprocessing doesn't support get_feature_names_out
            num_features = self.model.n_features_in_
            self.feature_names_ = [f"feature_{i}" for i in range(num_features)]

    # def configure(self, hyperparams):
    #     self.hyperparams = hyperparams
    #     self.model = RandomForestClassifier(
    #         **self.hyperparams
    #     )

    # return super().configure(hyperparams)


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
        top_k: int = 100,
        tuned_params=None,
    ):
        super().__init__(random_state=random_state, logger=logger, top_k=top_k)

        self.logger.log_check("Defining XGBoost...")
        self.use_scaling = False
        # self.top_k = top_k

        if tuned_params:
            self.model = XGBClassifier(**tuned_params)
        else:
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
        return self.pipeline.named_steps["model"]

    def set_model(self, model: XGBClassifier):
        self.model = model

    @timeit(process_name="XGB Fit")
    def fit(self, X_train, y_train, X_val, y_val):

        # 1️⃣ Fit preprocessing & transform both sets
        # This gives us the 273 columns (or top_k columns)
        X_train_ready = self.pipeline[:-1].fit_transform(X_train, y_train)
        X_val_ready = self.pipeline[:-1].transform(X_val)

        # 2️⃣ Access the model step directly and fit it
        # We use named_steps["model"] to call the XGBoost .fit() specifically
        self.model.fit(
            X_train_ready,
            y_train,
            eval_set=[(X_val_ready, y_val)],
            verbose=False,
        )
        self.pipeline.steps[-1] = ("model", self.model)

        # self.model = self.pipeline.named_steps["model"]

        # 3️⃣ Save feature names (now that Step 1 is done, these are available)
        try:
            self.feature_names_ = self.pipeline[:-1].get_feature_names_out().tolist()
        except Exception as e:
            self.feature_names_ = [
                f"feature_{i}" for i in range(X_train_ready.shape[1])
            ]

    # def configure(self, hyperparams):
    #     """
    #     Update the model with tuned hyperparameters and refresh the pipeline.
    #     """
    #     self.hyperparams = hyperparams

    #     # 1. Re-instantiate the model with new params
    #     self.model = XGBClassifier(**self.hyperparams)

    #     # 2. Re-build the pipeline so 'named_steps["model"]' points to the new instance
    #     self.set_up_pipeline()

    #     self.logger.log_result(f"Model re-configured with: {self.hyperparams}")


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, X):
        return self.model(X).squeeze(1)  # 👈 IMPORTANT


class NNWrapper(ModelWrapperBase):
    def __init__(
        self,
        random_state=DEF_RANDOM_STATE,
        logger=DEF_NOTEBOOK_LOGGER,
        top_k=100,
        tuned_params=None,
    ):
        super().__init__(random_state=random_state, logger=logger, top_k=top_k)

        self.use_scaling = True

        if tuned_params:
            self.model = NeuralNetClassifier(**tuned_params)
        else:
            self.model = NeuralNetClassifier(
                SimpleNN,
                module__input_dim=top_k,
                max_epochs=100,
                lr=0.001,
                batch_size=64,
                optimizer=torch.optim.Adam,
                criterion=torch.nn.BCEWithLogitsLoss,
                callbacks=[EarlyStopping(patience=10)],
                device="cuda" if torch.cuda.is_available() else "cpu",
                iterator_train__shuffle=True,
                # dtype=torch.float32,  # 🔥 ADD THIS
            )

        self.set_up_pipeline()

    def get_model(self):
        return self.model

    @timeit(process_name="NN Fit")
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # Convert target to numpy
        y_train_np = y_train.to_numpy().astype(np.float32)

        # 1️⃣ Fit preprocessing on training data (all steps except NN)
        X_train_transformed = self.pipeline[:-1].fit_transform(X_train, y_train_np)
        X_train_transformed = X_train_transformed.astype(np.float32)

        # Save only the top_k feature names after preprocessing
        # Use the shape of transformed X to get top-k
        top_k_used = X_train_transformed.shape[1]
        self.feature_names_ = X_train.columns.tolist()[:top_k_used]

        # 2️⃣ Update input_dim for NN
        input_dim = X_train_transformed.shape[1]
        self.model.set_params(module__input_dim=input_dim)

        # 3️⃣ Handle validation if provided
        if X_val is not None and y_val is not None:
            y_val_np = y_val.to_numpy().astype(np.float32)
            X_val_transformed = self.pipeline[:-1].transform(X_val)
            X_val_transformed = X_val_transformed.astype(np.float32)

            valid_ds = Dataset(X_val_transformed, y_val_np)
            self.model.set_params(train_split=predefined_split(valid_ds))

        # 4️⃣ Fit ONLY the neural net
        self.model.fit(X_train_transformed, y_train_np)
        self.pipeline.steps[-1] = ("model", self.model)

    # def configure(self, hyperparams):
    #     self.hyperparams = hyperparams
    #     self.model = NeuralNetClassifier(**self.hyperparams)


class ModelWrapperFactory:
    @staticmethod
    def create(
        model_type: SupportedModel,
        random_state: int,
        top_k: int,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
        scale_pos_weight=None,
        tuned_hyperparams: dict = None,
    ) -> ModelWrapperBase:
        if model_type == "RF":
            return RFWrapper(
                random_state, logger=logger, top_k=top_k, tuned_params=tuned_hyperparams
            )
        elif model_type == "XGB":
            return XGBWrapper(
                random_state,
                logger=logger,
                scale_pos_weight=scale_pos_weight,
                top_k=top_k,
                tuned_params=tuned_hyperparams,
            )
        elif model_type == "NN":
            return NNWrapper(
                random_state=random_state,
                logger=logger,
                top_k=top_k,
                tuned_params=tuned_hyperparams,
            )
        else:
            raise ValueError(f"Unexpected value: {model_type=}")
