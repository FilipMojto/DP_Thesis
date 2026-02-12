from joblib import Memory
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import FunctionTransformer, Pipeline
from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.config import DEF_RANDOM_STATE
from src_code.ml_pipeline.preprocessing.transform import build_transformer
from sklearn.preprocessing import StandardScaler

from src_code.ml_pipeline.training.constants import DEF_TOP_K


def to_float32(X):
    return X.astype(np.float32)

class PipelineBuilder:

    @staticmethod
    def build(
        model: BaseEstimator,
        logger: MyLogger,
        memory: Memory = None,
        random_state: int = DEF_RANDOM_STATE,
        top_k: int = DEF_TOP_K,
        use_scaling: bool = False,
    ):
        steps = [
            (
                "prep",
                build_transformer(random_state=random_state, logger=logger),
            )
        ]

        if use_scaling:
            steps.append(("scale", StandardScaler()))

        if top_k is not None:
            steps.append(("select", SelectKBest(score_func=f_classif, k=top_k)))


        steps.append(("to_float32", FunctionTransformer(lambda X: X.astype(np.float32)))),

        steps.append(("model", model))

        return Pipeline(steps, memory=memory) if memory else Pipeline(steps)
