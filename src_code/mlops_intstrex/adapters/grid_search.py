from sklearn.model_selection import GridSearchCV, ParameterGrid

from src_code.mlops_intstrex.joblib_progress import joblib_progress
from src_code.mlops_intstrex.reporters.progress_reporter import ProgressReporter


# def fit_with_progress(grid_search, X, y, reporter: ProgressReporter):

#     # n_candidates = len(grid_search.param_grid)
#     # # total = n_candidates * grid_search.cv
#     # total = len(ParameterGrid(grid_search.param_grid)) * grid_search.cv
#     n_candidates = len(ParameterGrid(grid_search.param_grid))
#     total = n_candidates * grid_search.cv

#     reporter.start(total, "GridSearchCV")

#     with joblib_progress(reporter, total_tasks=total):
#         grid_search.fit(X, y)

#     reporter.close()

class GridSearchAdapter:
    def __init__(self, grid_search: GridSearchCV):
        self.grid_search = grid_search
        # Pre-calculate total based on internal knowledge of GridSearchCV
        self.n_combos = len(ParameterGrid(grid_search.param_grid))
        self.total = self.n_combos * grid_search.cv

    def execute(self, reporter: ProgressReporter, X, y):
        reporter.start(self.total, "GridSearchCV")
        with joblib_progress(reporter, total_tasks=self.total):
            # self.grid_search.set_params()
            self.grid_search.fit(X, y)
        reporter.close()
        return self.grid_search