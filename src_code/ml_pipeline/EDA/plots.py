
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.EDA.utils import FeatureCategories
from src_code.ml_pipeline.EDA.utils import FeatureCategories
from src_code.ml_pipeline.scripts.train import RANDOM_STATE


def plot_label_distribution(df: pd.DataFrame, logger: MyLogger):
    logger.log_check("Checking Label Distribution...")


    # Plot
    sns.countplot(x='label', data=df)
    plt.title("Bug Label Distribution")
    plt.show()

    # Raw counts
    label_counts = df["label"].value_counts()
    logger.log_result(f"Label counts: {[(int(k), int(v)) for k, v in label_counts.items()]}")

    # Proportions
    label_props = df["label"].value_counts(normalize=True)
    logger.log_result(f"Label proportions: {[(int(k), round(float(v), 4)) for k, v in label_props.items()]}")


def plot_repo_distribution(df: pd.DataFrame, logger: MyLogger):
    logger.log_check("Checking commits per repository...")

    repo_counts = df['repo'].value_counts()

    # Plot
    plt.figure(figsize=(10,5))
    sns.barplot(x=repo_counts.index, y=repo_counts.values)
    plt.title("Commits per Repository")
    plt.ylabel("Number of Commits")
    plt.xticks(rotation=45)
    plt.show()

    # Raw counts (one-line log)
    logger.log_result(
        f"Repo commit counts: {[(repo, int(count)) for repo, count in repo_counts.items()]}"
    )

    # Proportions (one-line log)
    repo_props = repo_counts / len(df)
    logger.log_result(
        f"Repo commit proportions: {[(repo, round(float(prop), 4)) for repo, prop in repo_props.items()]}"
    )


def plot_num_feature_distributions(df: pd.DataFrame, logger: MyLogger, feature_ctgs):
    logger.log_check("Checking Numeric Feature Distribution Shape (NAME, SKEW, KURT, MEAN, STD)...")
    # num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # num_cols.remove('label')  # exclude target

    # embedding_cols = [c for c in num_cols if is_embedding_column(c)]
    # other_numeric_cols = [c for c in num_cols if not is_embedding_column(c)]

    summary = []

    for col in feature_ctgs.structural_cols:
        skew = round(stats.skew(df[col].dropna()), 3)
        kurt = round(stats.kurtosis(df[col].dropna()), 3)
        mean = round(df[col].mean(), 3)
        std = round(df[col].std(), 3)

        summary.append((col, skew, kurt, mean, std))

    # Log as single line
    logger.log_result(f"Numeric distributions summary: {summary}")

    # Optional: still show hist plots for visual inspection
    df[feature_ctgs.structural_cols].hist(bins=50, figsize=(20, 15))
    plt.suptitle("Numeric Feature Distributions")
    plt.show()


def plot_embedding_distributions(df: pd.DataFrame, feature_ctgs: FeatureCategories, logger: MyLogger):
    logger.log_check("Checking Embedding Feature Distributions...")

    def sample_embeddings(embedding_cols, logger: MyLogger, ratio=0.05, random_state=RANDOM_STATE):
        n_total = len(embedding_cols)
        n_sample = max(1, int(n_total * ratio))
        logger.log_result(f"Sampling {n_sample} out of {n_total} embedding columns for plotting.")

        rng = np.random.default_rng(random_state)
        return rng.choice(embedding_cols, size=n_sample, replace=False).tolist()

    sampled_emb_cols = sample_embeddings(feature_ctgs.embedding_cols, ratio=0.05, logger=logger)

    df[sampled_emb_cols].hist(bins=50, figsize=(20, 15))
    plt.suptitle(
        f"Embedding Distributions (Random {len(sampled_emb_cols)}/{len(feature_ctgs.embedding_cols)})"
    )
    plt.show()


def plot_file_types_distributions(df: pd.DataFrame, logger: MyLogger):
    logger.log_check("Checking File Extension Distribution...")

    # Compute extensions
    df['ext'] = df['filepath'].str.split('.').str[-1]

    # Counts
    ext_counts = df['ext'].value_counts()
    logger.log_result(f"Extension counts: {ext_counts.to_dict()}")

    # Proportions
    ext_props = (ext_counts / len(df)).round(4)
    logger.log_result(f"Extension proportions: {ext_props.to_dict()}")
    # Plot
    plt.figure(figsize=(8,4))
    sns.barplot(x=ext_counts.index, y=ext_counts.values)
    plt.title("File Extension Distribution")
    plt.ylabel("Count")
    plt.show()