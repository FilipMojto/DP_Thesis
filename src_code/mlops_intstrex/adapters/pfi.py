from sklearn.inspection import permutation_importance
from joblib import parallel_backend

from src_code.ml_pipeline.config import DEF_RANDOM_STATE
from src_code.ml_pipeline.utils import get_n_jobs
from src_code.mlops_intstrex.reporters.progress_reporter import ProgressReporter


# def permutation_importance_with_progress(
#     model, X, y, reporter, n_repeats=10
# ):

#     total = X.shape[1] * n_repeats
#     reporter.start(total, "Permutation Importance")

#     def wrapped_scorer(est, X, y):
#         reporter.advance()
#         return est.score(X, y)

#     result = permutation_importance(
#         model,
#         X,
#         y,
#         n_repeats=n_repeats,
#         scoring=wrapped_scorer
#     )


#     reporter.close()
#     return result
class PermutationImportanceAdapter:
    def __init__(
        self, model, n_repeats=10, reserve_cores=2, random_state=DEF_RANDOM_STATE
    ):
        self.model = model
        self.n_repeats = n_repeats
        self.n_jobs = get_n_jobs(
            reserve=reserve_cores
        )  # <--- Re-enabled parallel processing
        self.random_state = random_state

    # def execute(self, reporter: ProgressReporter, X, y):
    #     total = X.shape[1] * self.n_repeats
    #     reporter.start(total, "Permutation Importance")

    #     def wrapped_scorer(est, X_val, y_val):
    #         reporter.advance()
    #         return est.score(X_val, y_val)

    #     result = permutation_importance(
    #         self.model,
    #         X,
    #         y,
    #         n_repeats=self.n_repeats,
    #         scoring=wrapped_scorer,
    #         n_jobs=self.n_jobs,
    #         random_state=self.random_state,
    #     )
    #     reporter.close()
    #     return result
    def execute(self, reporter: ProgressReporter, X, y):
        total = X.shape[1] * self.n_repeats
        reporter.start(total, "Permutation Importance")

        def wrapped_scorer(est, X_val, y_val):
            reporter.advance()
            return est.score(X_val, y_val)

        # Force joblib to use threads instead of separate processes
        # This allows the 'reporter' object to be shared across workers
        with parallel_backend('threading', n_jobs=self.n_jobs):
            result = permutation_importance(
                self.model,
                X,
                y,
                n_repeats=self.n_repeats,
                scoring=wrapped_scorer,
                random_state=self.random_state,
            )
            
        reporter.close()
        return result
