
from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from notebooks.constants import NUMERIC_FEATURES
from notebooks.logging_config import MyLogger
from src_code.utils.utils import is_embedding_column, is_engineered, is_tfidf_vectorized


def log_general_overview(df: pd.DataFrame, logger: MyLogger):
    # log.info("[EDA CHECK] Checking the general overview...")
    logger.log_check("Checking the general overview...", print_to_console=True)
    # print(f"Dataset shape: {df.shape}")
    # log.info(f"[EDA RESULT] Dataset shape: {df.shape}")
    logger.log_result(f"Dataset shape: {df.shape}", print_to_console=True)

    # First few rows
    df.head()

    # Column types
    # df_dtypes = df.dtypes
    # Convert dtypes to list of tuples
    dtypes_list = [(col, str(dtype)) for col, dtype in df.dtypes.items()]

    # Log it
    # log.info(f"[EDA RESULT] Dtypes: {dtypes_list}")
    logger.log_result(f"Dtypes: {dtypes_list}", print_to_console=True)


def log_missing_values(df: pd.DataFrame, logger: MyLogger):
    # log.info("[EDA CHECK] Checking missing values...") 
    logger.log_check("Checking missing values...", print_to_console=True)

    # Check missing values
    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0]

    # Convert to list of tuples for compact logging
    missing_list = [(col, int(count)) for col, count in missing_nonzero.items()]

    # Log
    # log.info(f"[EDA RESULT] Missing values (only non-zero): {missing_list}")
    logger.log_result(f"Missing values (only non-zero): {missing_list}", print_to_console=True)

def log_numeric_features_with_negatives(df: pd.DataFrame, logger: MyLogger):
    logger.log_check("Checking negative numeric features...")

    negatives = []

    for col in NUMERIC_FEATURES:
        # Calculate the total number of rows (non-NaN count) in the column
        total_count = df[col].count()

        # Calculate the number of negative values in the column
        negative_count = (df[col] < 0).sum()

        # Calculate the proportion of negative values
        if total_count > 0:
            proportion = negative_count / total_count
        else:
            proportion = 0  # Handle case where the column is entirely NaN or empty

        if proportion:
            logger.log_result(
                f"Negative Feature: **{col}** - Count of negative values: {negative_count} - Proportion of negative values: **{proportion:.4f}"
            )
            # log_result(f"Total entries (non-NaN): {total_count}")
            # log_result(f"Count of negative values: {negative_count}")
            # log_result(f"Proportion of negative values: **{proportion:.4f}**")

    # log_result(f"Found following negative features: {negatives}")

    for negative in negatives:
        print(df[negative].describe())

    logger.log_result(f"Found following negative features: {negatives}")


@dataclass
class NumFeatureSets:
    numeric_cols: list
    engineered_cols: list
    embedding_cols: list
    tfidf_vectorized_cols: list
    # limit_features: list = None

    def all_numeric_cols(self):
        return self.embedding_cols + self.tfidf_vectorized_cols + self.numeric_cols

    @staticmethod
    def extract_features(df: pd.DataFrame, logger: MyLogger, limit_features: list = None):
      
        logger.log_check("Checking numeric features...")
        num_cols = df.select_dtypes(include=[np.number, np.bool]).columns.tolist()
        num_cols.remove('label')  # exclude target

        embedding_cols = [c for c in num_cols if is_embedding_column(c)]
        tfidf_vectorized_cols = [c for c in num_cols if is_tfidf_vectorized(c)]
        engineered_cols = [c for c in num_cols if is_engineered(c)]

        numeric_cols = [c for c in num_cols if not is_embedding_column(c) and not is_tfidf_vectorized(c) and not is_engineered(c)]

        # limit features for EDA
        if limit_features is not None:
            # lets also check if the specified features are actually in the dataframe
            for f in limit_features:
                if f not in df.columns:
                    raise ValueError(f"Specified feature '{f}' not found in dataframe columns.")
                
            embedding_cols = [f for f in embedding_cols if f in limit_features]
            tfidf_vectorized_cols = [f for f in tfidf_vectorized_cols if f in limit_features]
            engineered_cols = [f for f in engineered_cols if f in limit_features]
            numeric_cols = [f for f in numeric_cols if f in limit_features]


        logger.log_result(f"Embeddings: {embedding_cols}")
        logger.log_result(f"Size: {len(embedding_cols)}")
        logger.log_result("")
        logger.log_result(f"tfidf_vectorized: {tfidf_vectorized_cols}")
        logger.log_result(f"Size: {len(tfidf_vectorized_cols)}")
        logger.log_result("")

        logger.log_result(f"Structural: {numeric_cols}")
        logger.log_result(f"Engineered: {engineered_cols}")
        logger.log_result(f"Size: {len(numeric_cols)}")

        return NumFeatureSets(
            numeric_cols=numeric_cols,
            engineered_cols=engineered_cols,
            embedding_cols=embedding_cols,
            tfidf_vectorized_cols=tfidf_vectorized_cols,
        )


def log_empty_content_rows(df: pd.DataFrame, logger: MyLogger):
    # --- Empty content rows ---
    empty_content = int((df['content'].str.len() == 0).sum())
    logger.log_result(f"Empty content rows: {empty_content}")


def log_negative_time_since_last_change(df: pd.DataFrame, logger: MyLogger):
    logger.log_check("Checking Negative time_since_last_change per Repository...")

    # neg_times = df[df['time_since_last_change'] < 0]
    # neg_times['repo'].value_counts()
    neg_times = df[df['time_since_last_change'] < 0]
    neg_repo_counts = neg_times['repo'].value_counts()
    neg_repo_props = neg_repo_counts / len(df)

    for repo in neg_repo_counts.index:
        count = neg_repo_counts[repo]
        prop = neg_repo_props[repo]
        logger.log_result(f"{repo}: count={count}, proportion={prop:.3%}")
    sns.countplot(x='repo', data=neg_times)
    plt.title("Negative time_since_last_change by Repository")
    plt.show()


# def limit_features(df: pd.DataFrame, features: list, logger: MyLogger):
#     # lets also check if the specified features are actually in the dataframe
#         for f in limit_features:
#             if f not in df.columns:
#                 raise ValueError(f"Specified feature '{f}' not found in dataframe columns.")
            
#         features = [f for f in features if f in limit_features]
#         logger.log_check(f"Using specified features for boxplots: {features}")