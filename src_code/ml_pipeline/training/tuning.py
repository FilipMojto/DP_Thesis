import os
from joblib import Memory
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from abc import ABC, abstractmethod
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from notebooks.logging_config import MyLogger
from src_code.config import SupportedModel
from src_code.ml_pipeline.builders import PipelineBuilder
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER, DEF_RANDOM_STATE
from src_code.ml_pipeline.resources import CoreConfig
from src_code.ml_pipeline.training.constants import DEF_TOP_K

from src_code.mlops_intstrex.adapters.grid_search import GridSearchAdapter
from src_code.mlops_intstrex.reporters.progress_reporter import ProgressReporter
from src_code.mlops_intstrex.reporters.tqdm_reporter import TqdmReporter
from src_code.utils.utils import timeit


DEF_CORE_CONFIG = CoreConfig(reserve_cores=4, mode="all")


def to_numeric_df(X):
    # If X is already a DataFrame, we preserve index/columns
    if isinstance(X, pd.DataFrame):
        return X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # If it's a NumPy array (likely from the previous transformer),
    # convert it back to a DataFrame
    df = pd.DataFrame(X)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


def log_selected_features(grid_search, logger: MyLogger, top_n: int = 20):
    # 1. Get the best pipeline
    best_pipe = grid_search.best_estimator_

    # 2. Get the feature names from the 'prep' step
    # (Requires verbose_feature_names_out=True or unique names in ColumnTransformer)
    feature_names = best_pipe.named_steps["prep"].get_feature_names_out()

    # 3. Get the 'select' step and its scores/mask
    selector = best_pipe.named_steps["select"]
    scores = selector.scores_
    # This returns a boolean array (True if the feature was kept)
    mask = selector.get_support()

    # 4. Map names to scores and filter by selection
    selected_features = []
    for i, is_selected in enumerate(mask):
        if is_selected:
            selected_features.append((feature_names[i], scores[i]))

    # 5. Sort by score descending
    selected_features.sort(key=lambda x: x[1], reverse=True)

    # 6. Log the results
    logger.log_result(f"Top {top_n} Selected Features:")
    for index, (name, score) in enumerate(selected_features[:top_n], start=1):
        logger.log_result(f" - [{index}/{len(selected_features)}] {name}: {score:.4f}")


cachedir = os.path.join(os.getcwd(), ".pipeline_cache")
memory = Memory(location=cachedir, verbose=0)


class TuningWrapperBase(ABC):
    grid_search: GridSearchCV

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train,
        model: BaseEstimator,
        top_k: int,
        use_scaling: bool,
        param_grid: dict,
        logger: MyLogger,
        random_state: int = 42,
        reporter: ProgressReporter = None,
        core_config: CoreConfig = DEF_CORE_CONFIG,
        resource: str = "n_samples",
        max_resources: int = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.logger = logger
        self.random_state = random_state
        self.model = model
        self.param_grid = param_grid
        self.reporter = reporter or TqdmReporter()
        self.core_config = core_config
        self.top_k = top_k
        self.use_scaling = use_scaling

        self.pipeline = PipelineBuilder.build(
            model=self.model,
            logger=self.logger,
            memory=memory,
            random_state=self.random_state,
            top_k=self.top_k,
            use_scaling=self.use_scaling,
        )

        # self.mcc_scorer = make_scorer(matthews_corrcoef)
        self.f2_scorer = make_scorer(fbeta_score, beta=2)

        logger.log_check(
            f"Applying the following core config: {self.core_config.__str__()}"
        )
        self.grid_search = HalvingGridSearchCV(
            self.pipeline,
            param_grid=self.param_grid,
            factor=3,  # Min candidates to keep in each round
            resource=resource,
            scoring=self.f2_scorer,
            cv=5,
            n_jobs=self.core_config.n_jobs,
        )
        self.grid_search_adapter = GridSearchAdapter(self.grid_search)

    @abstractmethod
    def run_grid_search(self):
        pass

    def get_best_score(self):
        return self.grid_search.best_params_, self.grid_search.best_score_


class RFTuningWrapper(TuningWrapperBase):
    PARAM_GRID = {
        # "model__n_estimators": [100],
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2"],
    }

    def __init__(
        self,
        rf: RandomForestClassifier,
        X_train,
        y_train,
        logger: MyLogger,
        top_k: int = DEF_TOP_K,
        use_scaling: bool = False,
        random_state: int = DEF_RANDOM_STATE,
        reporter: ProgressReporter | None = None,
        core_config: CoreConfig = DEF_RANDOM_STATE,
    ):
        super().__init__(
            X_train=X_train,
            y_train=y_train,
            model=rf,
            param_grid=self.PARAM_GRID,
            logger=logger,
            random_state=random_state,
            reporter=reporter,
            core_config=core_config,
            use_scaling=use_scaling,
            top_k=top_k,
        )

        logger.log_check("Initializing RFTunning Wrapper object...")
        self.logger.log_result("Initialization completed.")

    @timeit(process_name="RF - Grid Search")
    def run_grid_search(self):
        self.grid_search_adapter.execute(self.reporter, self.X_train, self.y_train)


class XGBTuningWrapper(TuningWrapperBase):
    PARAM_GRID = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    def __init__(
        self,
        xgb: XGBClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        logger: MyLogger,
        top_k: int = DEF_TOP_K,
        use_scaling: bool = False,
        random_state: int = DEF_RANDOM_STATE,
        reporter: ProgressReporter | None = None,
        core_config: CoreConfig = DEF_CORE_CONFIG,
    ):
        # super().__init__(X_train, y_train, xgb, logger, random_state, reporter)
        super().__init__(
            # *args,
            # **kwargs,
            X_train=X_train,
            y_train=y_train,
            model=xgb,
            param_grid=self.PARAM_GRID,
            logger=logger,
            random_state=random_state,
            reporter=reporter,
            core_config=core_config,
            use_scaling=use_scaling,
            top_k=top_k,
        )
        logger.log_check("Initializing XGBTunning Wrapper object...")

    @timeit(process_name="XGB - Grid Search")
    def run_grid_search(self):
        self.grid_search_adapter.execute(self.reporter, self.X_train, self.y_train)


class NNTuningWrapper(TuningWrapperBase):
    PARAM_GRID = {
        "model__lr": [0.002, 0.0005],
        # "model__max_epochs": [20, 50],
        "model__module__hidden_units": [64, 128],
    }

    def __init__(
        self,
        X_train,
        y_train,
        model,
        logger,
        random_state=42,
        reporter=None,
        core_config=DEF_CORE_CONFIG,
        top_k=DEF_TOP_K,
        use_scaling=True,
    ):
        super().__init__(
            X_train=X_train,
            y_train=y_train,
            model=model,
            param_grid=self.PARAM_GRID,
            logger=logger,
            random_state=random_state,
            reporter=reporter,
            core_config=core_config,
            use_scaling=use_scaling,
            top_k=top_k,
        )
        self.y_train = self.y_train.astype("float32")


    @timeit(process_name="NN - Grid Search")
    def run_grid_search(self):
        self.grid_search_adapter.execute(self.reporter, self.X_train, self.y_train)


class LRTuningWrapper(TuningWrapperBase):
    # PARAM_GRID = {
    #     "model__C": [0.01, 0.1, 1.0, 10.0],  # Regularization strength
    #     "model__penalty": ["l1", "l2"],
    #     "model__solver": ["liblinear", "saga"],  # compatible with l1/l2
    #     "model__max_iter": [500, 1000],
    # }
    # PARAM_GRID = {
    #     "model__C": [0.01, 0.1, 1.0, 10.0],
    #     "model__solver": ["saga"],  # saga supports all types
    #     "model__l1_ratio": [None, 0.0, 0.5, 1.0],  # None for pure l2
    #     "model__max_iter": [500, 1000]
    # }
    PARAM_GRID = {
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__max_iter": [500, 1000],
    }

    def __init__(
        self,
        lr: LogisticRegression,
        X_train,
        y_train,
        logger: MyLogger,
        top_k: int = DEF_TOP_K,
        use_scaling: bool = True,  # LR benefits from scaling
        random_state: int = DEF_RANDOM_STATE,
        reporter=None,
        core_config=None,
    ):
        super().__init__(
            X_train=X_train,
            y_train=y_train,
            model=lr,
            param_grid=self.PARAM_GRID,
            logger=logger,
            random_state=random_state,
            reporter=reporter,
            core_config=core_config or DEF_RANDOM_STATE,
            use_scaling=use_scaling,
            top_k=top_k,
        )

        logger.log_check("Initializing Logistic Regression Tuning Wrapper...")
        self.logger.log_result("Initialization completed.")

    @timeit(process_name="LR - Grid Search")
    def run_grid_search(self):
        self.grid_search_adapter.execute(self.reporter, self.X_train, self.y_train)

class ModelTuningFactory:
    @staticmethod
    def create(
        model_type: SupportedModel,
        model,
        X_train: pd.DataFrame,
        y_train,
        # n_workers: int,
        # reserve_cores: int,
        core_config=DEF_CORE_CONFIG,
        random_state: int = DEF_RANDOM_STATE,
        # X_val=None,
        # y_val=None,
        logger=DEF_NOTEBOOK_LOGGER,
    ):
        model_type = model_type.upper()

        if model_type == "RF":
            return RFTuningWrapper(
                rf=model,
                X_train=X_train,
                y_train=y_train,
                logger=logger,
                random_state=random_state,
                core_config=core_config,
            )
        elif model_type == "XGB":
            return XGBTuningWrapper(
                xgb=model,
                X_train=X_train,
                y_train=y_train,
                # X_val=X_val,
                # y_val=y_val,
                logger=logger,
                random_state=random_state,
                # workers_n=n_workers,
                # res
                core_config=core_config,
            )
        elif model_type == "NN":
            return NNTuningWrapper(
                model=model,
                X_train=X_train,
                y_train=y_train,
                logger=logger,
                random_state=random_state,
                core_config=core_config,
            )
        elif model_type == "LR":
            return LRTuningWrapper(
                lr=model,
                X_train=X_train,
                y_train=y_train,
                logger=logger,
                random_state=random_state,
                core_config=core_config,
            )
        else:
            raise ValueError(f"Invalid model_type argumnent: {model_type}")


# import argparse

if __name__ == "__main__":
    ...
    # args = argparse.ArgumentParser(description="Model Tuning")
    # args.add_argument(
    #     "--model",
    #     type=str,
    #     required=True,
    #     choices=get_args(SupportedModel),
    #     help="Type of model to tune: 'rf' for Random Forest, 'xgb' for XGBoost",
    # )
    # script_logger = MyLogger(
    #     label="TUNING",
    #     section_name="TUNING LOGGER SCRIPT",
    #     file_log_path=LOG_DIR / "tuning_log.log",
    # )
    # script_logger.start_session(session_id=random.randint(1000, 9999))
    # parsed_args = args.parse_args()

    # dataset: pd.DataFrame = load_df(
    #     df_file_path=ENGINEERING_MAPPINGS["train"]["output"]
    # )
    # X_train = dataset.drop(columns=[TARGET])
    # y_train = dataset[TARGET]

    # # X_train = drop_cols(df=X_train, cols=DROP_COLS, logger=script_logger)

    # model_type = parsed_args.model.lower()

    # model_wrapper = ModelWrapperFactory.create(model_type=model_type, random_state=42)[
    #     0
    # ]
    # model = model_wrapper.get_model()

    # tuner = ModelTuningFactory.create(
    #     model_type=model_type, model=model, X_train=X_train, y_train=y_train
    # )

    # best_params, best_score = tuner.run_grid_search()
    # script_logger.log_result(
    #     f"Tuning completed for {model_type.upper()}. Best Params: {best_params}, Best Score: {best_score}"
    # )

