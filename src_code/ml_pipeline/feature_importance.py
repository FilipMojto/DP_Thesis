import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.utils import get_n_jobs


class PFIWrapper:
    def __init__(
        self,
        model: BaseEstimator,
        # X_test,
        # y_test,
        random_state: int,
        # X_val=None,
        # y_val=None,
        logger: MyLogger = DEF_NOTEBOOK_LOGGER,
        reserve_cores: int = 2,
    ):
        self.logger = logger
        self.logger.log_check("Initializing PFI wrapper..")

        # self.perm = permutation_importance(
        self.model = model
        # self.X_test = X_test
        # self.y_test = y_test
        # self.X_val = X_val
        # self.y_val = y_val
        self.n_repeats = 2
        self.random_state = random_state
        self.n_jobs = get_n_jobs(
            reserve=reserve_cores
        )  # <--- Re-enabled parallel processing
        # )

        self.logger.log_check("Initialization done.")

    def run_PFI(self, X_test, y_test, top_k: int = 10):
        self.logger.log_check("Starting PFI...")
        self.perm = permutation_importance(
            self.model,
            X_test,
            y_test,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,  # <--- Re-enabled parallel processing
        )

        self.logger.log_result("PFI completed.")

    # def calc_importances(self):
        self.logger.log_check("Calculating PFI importances...")
        self.importances = pd.Series(
            self.perm.importances_mean,  # Retrieves the average importance score
            # (the average drop in model performance)
            # calculated across the n_repeats=2 runs
            # for each feature.
            # index=model.named_steps["preprocess"].get_feature_names_out()\
            index=X_test.columns,
            # This is a crucial step for pipelines. After the ColumnTransformer
            # ("preprocess") has run (including PCA and any other steps), the feature
            #  names are transformed (e.g., code_emb_0 becomes embed__pca__0). This
            # method retrieves the correct, final feature names that the model actually used.
        ).sort_values(ascending=False)
        self.logger.log_result("Calculation complete.")

    # def get_importances(self, top_k: int = 10):
        # self.importances = pd.Series(
        #     self.perm.importances_mean, # Retrieves the average importance score
        #                             # (the average drop in model performance)
        #                             # calculated across the n_repeats=2 runs
        #                             # for each feature.
        #     # index=model.named_steps["preprocess"].get_feature_names_out()\
        #     index=self.X_test.columns

        #     # This is a crucial step for pipelines. After the ColumnTransformer
        #     # ("preprocess") has run (including PCA and any other steps), the feature
        #     #  names are transformed (e.g., code_emb_0 becomes embed__pca__0). This
        #     # method retrieves the correct, final feature names that the model actually used.
        # ).sort_values(ascending=False)

        self.logger.log_result(f"Top {top_k} PFI features:")
        top_importances = self.importances.head(top_k).items()
        for i, (feature, value) in enumerate(top_importances, start=1):
            self.logger.log_result(f"{i:2d}. {feature}: {value:.6f}")

        return top_importances

    def refine_features(self, X_train, X_test, X_val = None, threshold: float = 0.0001):
        self.logger.log_check("Refining features based on best PFI importances...")
        # threshold = 0.0001 # Or use 0.0 to be more inclusive
        top_features = self.importances[self.importances > threshold].index.tolist()

        # Filter your training and testing sets
        X_train_filtered = X_train[top_features]
        X_test_filtered = X_test[top_features]

        if X_val is not None:
            X_val_filtered = X_val[top_features]
            # return X_train_filtered, X_test_filtered, X_val_filtered

        # df_test = pd.read_feather(ENGINEERING_MAPPINGS['test']['output'])
        # top_filter = top_features.copy()
        # top_filter.append('label')
        # print(top_filter)
        # df_test = df_test[top_filter]
        # df_test.to_feather("lala.feather")

        self.logger.log_result(
            f"Reduced feature count from {len(self.importances)} to {len(top_features)}"
        )

        self.logger.log_result(f"Features retained: {top_features}")
        return X_train_filtered, X_test_filtered, X_val_filtered if X_val is not None else None
