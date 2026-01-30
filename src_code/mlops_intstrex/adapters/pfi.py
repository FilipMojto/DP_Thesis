from sklearn.inspection import permutation_importance

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
    def __init__(self, model, n_repeats=10):
        self.model = model
        self.n_repeats = n_repeats

    def execute(self, reporter: ProgressReporter, X, y):
        total = X.shape[1] * self.n_repeats
        reporter.start(total, "Permutation Importance")
        
        def wrapped_scorer(est, X_val, y_val):
            reporter.advance()
            return est.score(X_val, y_val)

        result = permutation_importance(
            self.model, X, y, n_repeats=self.n_repeats, scoring=wrapped_scorer
        )
        reporter.close()
        return result