from abc import abstractmethod
from collections import Counter
from typing import Dict, List
import numpy as np
from pyparsing import ABC
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression
import torch
from xgboost import XGBClassifier
import torch.nn as nn
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from notebooks.logging_config import MyLogger
from src_code.config import SupportedModel
from src_code.ml_pipeline.builders import PipelineBuilder
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER, DEF_RANDOM_STATE
from src_code.utils.utils import timeit


DEF_N_JOBS = 1  # 🔴 IMPORTANT – consistent across models


def log_native_feature_importance(
    model,
    feature_names: list[str],
    logger: MyLogger,
    top_n: int = 20,
    title: str = "Model-native feature importance",
):
    if not hasattr(model, "feature_importances_"):
        logger.log_result(f"{title}: not available for this model.")
        return

    importances = np.asarray(model.feature_importances_, dtype=float)

    if len(importances) != len(feature_names):
        logger.log_result(
            f"{title}: length mismatch between importances ({len(importances)}) "
            f"and feature names ({len(feature_names)})."
        )
        return

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.log_result(f"{title} | Top {top_n}:")
    for i, (name, score) in enumerate(ranked[:top_n], start=1):
        logger.log_result(f" - [{i}/{len(ranked)}] {name}: {score:.6f}")


class ModelWrapperBase(ABC):
    def __init__(
        self,
        top_k: int,
        available_cols: List[str],
        random_state: int = DEF_RANDOM_STATE,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
    ):
        self.logger = logger
        self.model: BaseEstimator = None
        self.random_state = random_state
        self.available_cols = available_cols

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

    def transform(self, X):
        return self.pipeline[:-1].transform(X).astype(np.float32)

    def set_up_pipeline(self):
        self.pipeline = PipelineBuilder.build(
            model=self.model,
            logger=self.logger,
            random_state=self.random_state,
            top_k=self.top_k,
            use_scaling=self.use_scaling,
            available_cols=self.available_cols,
        )


class RFWrapper(ModelWrapperBase):

    def __init__(
        self,
        random_state: int,
        available_cols: List[str],
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
        top_k: int = 100,
        tuned_params=None,
    ):
        super().__init__(
            random_state=random_state,
            logger=logger,
            top_k=top_k,
            available_cols=available_cols,
        )

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
                class_weight="balanced",  # <-- ADD THIS
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
        # 1️⃣ Fit the entire pipeline
        # (RF doesn't need an eval_set, so we can do this in one shot)
        # self.pipeline.fit(X_train, y_train)
        X_train_ready = self.pipeline[:-1].fit_transform(X_train, y_train)

        self.model.fit(X_train_ready, y_train)
        self.pipeline.steps[-1] = ("model", self.model)

        # 2️⃣ Sync the internal model referenc

        # 3️⃣ Save the CORRECT processed feature names
        try:
            # This pulls names from the pipeline steps (encoding, selection, etc.)
            self.feature_names_ = self.pipeline[:-1].get_feature_names_out().tolist()
            log_native_feature_importance(
                model=self.model, feature_names=self.feature_names_, logger=self.logger
            )

        except Exception as e:
            # Fallback if preprocessing doesn't support get_feature_names_out
            num_features = self.model.n_features_in_
            self.feature_names_ = [f"feature_{i}" for i in range(num_features)]


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
        available_cols: List[str],
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
        scale_pos_weight=None,
        top_k: int = 100,
        tuned_params=None,
    ):
        super().__init__(
            random_state=random_state,
            logger=logger,
            top_k=top_k,
            available_cols=available_cols,
        )

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
            log_native_feature_importance(
                model=self.model, feature_names=self.feature_names_, logger=self.logger
            )
        except Exception as e:
            self.feature_names_ = [
                f"feature_{i}" for i in range(X_train_ready.shape[1])
            ]


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_units=128, dropout=0.3):
        super().__init__()

        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, hidden_units),
        #     nn.BatchNorm1d(hidden_units),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),   # 🔥 now configurable
        #     nn.Linear(hidden_units, 1),
        # )
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Linear(hidden_units // 2, 1),
        )

        self.model.apply(init_weights)

    def forward(self, X):
        return self.model(X).squeeze(1)


class FloatTargetNeuralNetBinaryClassifier(NeuralNetBinaryClassifier):
    def fit(self, X, y, **fit_params):
        y = np.asarray(y, dtype=np.float32)
        return super().fit(X, y, **fit_params)

    def partial_fit(self, X, y, classes=None, **fit_params):
        y = np.asarray(y, dtype=np.float32)
        return super().partial_fit(X, y, classes=classes, **fit_params)


class NNWrapper(ModelWrapperBase):
    def __init__(
        self,
        available_cols: List[str],
        random_state=DEF_RANDOM_STATE,
        logger=DEF_NOTEBOOK_LOGGER,
        top_k=100,
        tuned_params=None,
    ):
        super().__init__(
            random_state=random_state,
            logger=logger,
            top_k=top_k,
            available_cols=available_cols,
        )

        self.use_scaling = True

        if tuned_params:
            self.model = FloatTargetNeuralNetBinaryClassifier(**tuned_params)
        else:
            self.model = FloatTargetNeuralNetBinaryClassifier(
                SimpleNN,
                module__input_dim=self.top_k,
                module__hidden_units=128,
                max_epochs=100,
                lr=0.002,
                batch_size=64,
                optimizer=torch.optim.Adam,
                criterion=torch.nn.BCEWithLogitsLoss,
                callbacks=[EarlyStopping(patience=10)],
                device="cuda" if torch.cuda.is_available() else "cpu",
                iterator_train__shuffle=True,
                # dtype=torch.float32,   # 🔥 THIS FIXES IT
            )

        self.set_up_pipeline()

    def get_model(self):
        return self.model

    @timeit(process_name="NN Fit")
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # Convert target to numpy
        y_train_np = y_train.to_numpy().astype(np.float32)

        # pos_weight = torch.tensor([y_train_np.shape[0] / (2 * y_train_np.sum())], dtype=torch.float32)
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # self.model.set_params(criterion=criterion)
        # Calculate your weight
        # pos_weight = torch.tensor([y_train_np.shape[0] / (2 * y_train_np.sum())], dtype=torch.float32)
        num_pos = y_train_np.sum()
        num_neg = len(y_train_np) - num_pos
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32)
        self.logger.log_result(f"Calculated pos_weight: {pos_weight}")

        # Set the parameter, not the object.
        # Note: This might still trigger an initialization error if the net is already "warm"
        self.model.set_params(criterion__pos_weight=pos_weight)

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


class LRWrapper(ModelWrapperBase):
    """
    Wrapper for LogisticRegression to integrate into the ML pipeline.
    """

    def __init__(
        self,
        available_cols: List[str],
        random_state: int = DEF_RANDOM_STATE,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
        top_k: int = 100,
        tuned_params: dict = None,
    ):
        super().__init__(
            top_k=top_k,
            random_state=random_state,
            logger=logger,
            available_cols=available_cols,
        )

        self.logger.log_check("Defining Logistic Regression model...")
        self.use_scaling = True  # LR benefits from feature scaling

        # Initialize model
        if tuned_params:
            self.model = LogisticRegression(**tuned_params)
        else:
            # self.model = LogisticRegression(
            #     penalty="l2",
            #     solver="liblinear",  # good default for small-medium datasets
            #     C=1.0,
            #     random_state=random_state,
            #     class_weight="balanced",
            #     max_iter=1000,
            # )
            self.model = LogisticRegression(
                solver="lbfgs",
                C=1.0,
                random_state=random_state,
                class_weight="balanced",
                max_iter=1000,
            )

        # Build pipeline
        self.set_up_pipeline()
        logger.log_result("Logistic Regression wrapper initialized.")

    def get_model(self):
        return self.model

    def set_model(self, lr: LogisticRegression):
        self.model = lr

    @timeit(process_name="LR Fit")
    def fit(self, X_train, y_train):
        """
        Fit the preprocessing pipeline + logistic regression model.
        """
        # Fit preprocessing steps first
        X_train_ready = self.pipeline[:-1].fit_transform(X_train, y_train)

        # Fit LR on transformed features
        self.model.fit(X_train_ready, y_train)
        self.pipeline.steps[-1] = ("model", self.model)

        # Store feature names
        try:
            self.feature_names_ = self.pipeline[:-1].get_feature_names_out().tolist()
        except Exception:
            num_features = self.model.n_features_in_
            self.feature_names_ = [f"feature_{i}" for i in range(num_features)]


class DummyWrapper(ModelWrapperBase):

    def __init__(
        self,
        strategy: str = "stratified",  # or "most_frequent"
        random_state: int = DEF_RANDOM_STATE,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
        top_k: int = None,  # not needed
    ):
        super().__init__(random_state=random_state, logger=logger, top_k=top_k)

        self.logger.log_check(f"Defining Dummy model: {strategy}")

        self.use_scaling = False

        self.model = DummyClassifier(strategy=strategy, random_state=random_state)

        self.set_up_pipeline()

        logger.log_result("Dummy model definition done.")

    def get_model(self):
        return self.model

    @timeit(process_name="Dummy Fit")
    def fit(self, X_train, y_train):
        # Same pattern as RF
        X_ready = self.pipeline[:-1].fit_transform(X_train, y_train)

        self.model.fit(X_ready, y_train)
        self.pipeline.steps[-1] = ("model", self.model)

        # Feature names (optional)
        try:
            self.feature_names_ = self.pipeline[:-1].get_feature_names_out().tolist()
        except:
            self.feature_names_ = [
                f"feature_{i}" for i in range(self.model.n_features_in_)
            ]


class VotingWrapper(ModelWrapperBase):

    def __init__(
        self,
        base_wrappers: Dict[
            str, ModelWrapperBase
        ],  # {"rf": RFWrapper, "xgb": XGBWrapper, ...}
        random_state: int,
        logger,
        available_cols: List[str],
        top_k: int = 100,
    ):
        super().__init__(
            top_k=top_k,
            random_state=random_state,
            logger=logger,
            available_cols=available_cols,
        )

        self.logger.log_check("Defining Voting Ensemble...")

        # Extract trained models
        estimators = [(name, wrapper.model) for name, wrapper in base_wrappers.items()]

        self.model = VotingClassifier(
            estimators=estimators, voting="soft", n_jobs=-1  # IMPORTANT
        )

        self.use_scaling = False  # depends on base models
        self.set_up_pipeline()

        self.logger.log_result("Voting ensemble initialized.")

    def get_model(self):
        return self.model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # For voting ensemble, we fit the ensemble on the outputs of base models.
        X_ready = self.pipeline[:-1].fit_transform(X_train, y_train)
        self.model.fit(X_ready, y_train)
        self.pipeline.steps[-1] = ("model", self.model)


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


class StackingWrapper(ModelWrapperBase):

    def __init__(
        self,
        base_wrappers: dict,
        random_state: int,
        logger,
        available_cols: List[str],
        top_k: int = 100,
    ):
        super().__init__(
            top_k=top_k,
            random_state=random_state,
            logger=logger,
            available_cols=available_cols,
        )

        self.logger.log_check("Defining Stacking Ensemble...")

        estimators = [(name, wrapper.model) for name, wrapper in base_wrappers.items()]

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_state
            ),
            stack_method="predict_proba",  # important for your use-case
            n_jobs=-1,
        )

        self.use_scaling = False
        self.set_up_pipeline()

        self.logger.log_result("Stacking ensemble initialized.")

    def get_model(self):
        return self.model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_ready = self.pipeline[:-1].fit_transform(X_train, y_train)
        self.model.fit(X_ready, y_train)
        self.pipeline.steps[-1] = ("model", self.model)


class ModelWrapperFactory:
    @staticmethod
    def create(
        model_type: SupportedModel,
        random_state: int,
        top_k: int,
        available_cols: List[str],
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
        scale_pos_weight=None,
        tuned_hyperparams: dict = None,
        base_wrappers: Dict[str, ModelWrapperBase] = None,  # for ensemble
    ) -> ModelWrapperBase:
        if model_type == "RF":
            return RFWrapper(
                random_state,
                logger=logger,
                top_k=top_k,
                tuned_params=tuned_hyperparams,
                available_cols=available_cols,
            )
        elif model_type == "XGB":
            return XGBWrapper(
                random_state,
                logger=logger,
                scale_pos_weight=scale_pos_weight,
                top_k=top_k,
                tuned_params=tuned_hyperparams,
                available_cols=available_cols,
            )
        elif model_type == "NN":
            return NNWrapper(
                random_state=random_state,
                logger=logger,
                top_k=top_k,
                tuned_params=tuned_hyperparams,
                available_cols=available_cols,
            )
        elif model_type == "LR":
            return LRWrapper(
                random_state=random_state,
                logger=logger,
                top_k=top_k,
                tuned_params=tuned_hyperparams,
                available_cols=available_cols,
            )
        elif model_type == "ENSEMBLE_VOTING":
            return VotingWrapper(
                base_wrappers=base_wrappers,
                random_state=random_state,
                logger=logger,
                top_k=top_k,
                available_cols=available_cols,
            )
        elif model_type == "DUMMY_STRATIFIED":
            return DummyWrapper(
                strategy="stratified",
                random_state=random_state,
                logger=logger,
            )
        elif model_type == "DUMMY_MOST_FREQUENT":
            return DummyWrapper(
                strategy="most_frequent",
                random_state=random_state,
                logger=logger,
            )
        elif model_type == "ENSEMBLE_STACKING":
            return StackingWrapper(
                base_wrappers=base_wrappers,
                random_state=random_state,
                logger=logger,
                top_k=top_k,
                available_cols=available_cols,
            )
        else:
            raise ValueError(f"Unexpected value: {model_type=}")
