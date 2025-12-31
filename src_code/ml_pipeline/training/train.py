import time
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.config import DEF_NOTEBOOK_LOGGER
from src_code.ml_pipeline.models import ModelWrapperBase


def split_train_test(
    df: pd.DataFrame,
    target: str,
    random_state: int,
    test_size: float,
    logger: MyLogger = DEF_NOTEBOOK_LOGGER,
):
    logger.log_check("Splitting df into train & test subsets...")
    n_rows = len(df)
    n_features = df.shape[1] - 1  # excluding target

    logger.log_result(f"Total rows before split: {n_rows}")
    logger.log_result(f"Feature count (X): {n_features}")
    logger.log_result(f"Target column: '{target}'")
    logger.log_result(f"Test size: {test_size:.2%}")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        # stratify=y, not required since the training subset of the original df is balanced
        random_state=random_state,
    )

    logger.log_result(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    logger.log_result("Splitting completed.")
    return X_train, X_test, y_train, y_test


def fit_rf(
    model: BaseEstimator,
    X_train,
    y_train,
    logger: MyLogger = DEF_NOTEBOOK_LOGGER,
):
    # This step trains the single, final model pipeline that is saved
    # in the 'model' variable and used for prediction and PFI.
    logger.log_check("Starting model fit...")

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    logger.log_result(f"Model fit completed. Time: {end - start:2f}")
    return model


def fit_xgb_with_es(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    logger: MyLogger = DEF_NOTEBOOK_LOGGER,
    use_early_stopping=True,
):
    logger.log_check("Starting XGBoost fit...")
    start = time.time()

    if use_early_stopping:
        # X_tr, X_val, y_tr, y_val = train_test_split(
        #     X_train,
        #     y_train,
        #     test_size=0.15,
        #     random_state=42,
        #     stratify=y_train,
        # )

        model.set_params(
            n_estimators=3000,
            learning_rate=0.05,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    end = time.time()
    logger.log_result(f"XGBoost fit completed. Time: {end - start:.2f}s")

    return model


def fit_model(model_type: str, model_wrapper: ModelWrapperBase, X_train, y_train, X_validate=None, y_validate=None):
    if model_type == "RF":
        model_wrapper.fit(X_train, y_train)
    elif model_type == "XGB":
        # model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        model_wrapper.fit(X_train, y_train, X_val=X_validate, y_val=y_validate)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model_wrapper


def check_single_infer(
    model: BaseEstimator,
    X_test,
    logger: MyLogger = DEF_NOTEBOOK_LOGGER,
):
    logger.log_check("Checking single model inference...")
    start_time = time.time()

    # This call runs the entire pipeline: Preprocessing (PCA) + Random Forest Prediction
    # The output is not needed, just the execution time.
    _ = model.predict(X_test)

    end_time = time.time()
    single_inference_duration = end_time - start_time
    logger.log_result("Inference done.")

    logger.log_result(
        f"Time for a single inference run on X_test ({len(X_test)} rows): {single_inference_duration:.2f} seconds"
    )

    # logger.log_result("")


# def perform_PFI(random_state: int, X_test, y_test, model: BaseEstimator):
#     X_test_small = X_test.sample(n=5000, random_state=random_state)
#     y_test_small = y_test.loc[X_test_small.index]


#     # The total number of tasks is N_features * n_repeats
#     # n_features = len(model.named_steps["preprocess"].get_feature_names_out())
#     n_features = X_test_small.shape[1]
#     total_tasks = n_features * 2

#     # with parallel_backend('loky', n_jobs=-1): # Use all cores
#         # with tqdm.tqdm(total=total_tasks, desc="PFI Permutations") as progress_bar:
#             # Wrap the function call in a helper that updates the progress bar
#             # This is a bit advanced but forces joblib to use the tqdm callback

#     # NOTE: In modern scikit-learn/joblib, simply setting the backend
#     # is often enough to show the progress. If not, this is the safest way:
#     perm = permutation_importance(
#         model,
#         X_test_small,
#         y_test_small,
#         n_repeats=2,
#         random_state=random_state,
#         n_jobs=6, # <--- Re-enabled parallel processing
#     )

#     importances = pd.Series(
#         perm.importances_mean, # Retrieves the average importance score
#                                 # (the average drop in model performance)
#                                 # calculated across the n_repeats=2 runs
#                                 # for each feature.
#         # index=model.named_steps["preprocess"].get_feature_names_out()\
#         index=X_test_small.columns

#         # This is a crucial step for pipelines. After the ColumnTransformer
#         # ("preprocess") has run (including PCA and any other steps), the feature
#         #  names are transformed (e.g., code_emb_0 becomes embed__pca__0). This
#         # method retrieves the correct, final feature names that the model actually used.
#     ).sort_values(ascending=False)

#     importances.head(20)
