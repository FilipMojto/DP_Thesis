from pathlib import Path
from typing import Iterable, Literal
from notebooks.constants import ENGINEERED_FEATURES
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.EDA.utils import FeatureCategories
from src_code.ml_pipeline.EDA.utils import FeatureCategories
from src_code.ml_pipeline.experimenting.utils import save_plt_image
from src_code.ml_pipeline.scripts.train import RANDOM_STATE


def plot_label_distribution(df: pd.DataFrame, logger: MyLogger):
    logger.log_check("Checking Label Distribution...")

    # Plot
    sns.countplot(x="label", data=df)
    plt.title("Bug Label Distribution")
    plt.show()

    # Raw counts
    label_counts = df["label"].value_counts()
    logger.log_result(
        f"Label counts: {[(int(k), int(v)) for k, v in label_counts.items()]}"
    )

    # Proportions
    label_props = df["label"].value_counts(normalize=True)
    logger.log_result(
        f"Label proportions: {[(int(k), round(float(v), 4)) for k, v in label_props.items()]}"
    )


def plot_repo_distribution(df: pd.DataFrame, logger: MyLogger):
    logger.log_check("Checking commits per repository...")

    repo_counts = df["repo"].value_counts()

    # Plot
    plt.figure(figsize=(10, 5))
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


def plot_num_feature_distributions(
    df: pd.DataFrame,
    logger: MyLogger,
    feature_ctgs: FeatureCategories,
    col_type: Literal['structural', 'embedding', 'vectorized'] = 'structural',
    experiment_dir: Path = None,
):
    def sample_cols(
        cols, logger: MyLogger, ratio=0.05, random_state=RANDOM_STATE
    ):
        n_total = len(cols)
        n_sample = max(1, int(n_total * ratio))
        logger.log_result(
            f"Sampling {n_sample} out of {n_total} embedding columns for plotting."
        )

        rng = np.random.default_rng(random_state)
        return rng.choice(cols, size=n_sample, replace=False).tolist()


    logger.log_check(
        "Checking Numeric Feature Distribution Shape (NAME, SKEW, KURT, MEAN, STD)..."
    )
    # num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # num_cols.remove('label')  # exclude target

    # embedding_cols = [c for c in num_cols if is_embedding_column(c)]
    # other_numeric_cols = [c for c in num_cols if not is_embedding_column(c)]

    summary = []
    # filtered_cols = [
    #     feature
    #     for feature in feature_ctgs.structural_cols
    #     if feature not in ENGINEERED_FEATURES
    # ]
    # filtered_cols = cols if cols else feature_ctgs.structural_cols

    match col_type:
        case 'structural':
            filtered_cols = feature_ctgs.structural_cols
        case 'embedding':
            # filtered_cols = feature_ctgs.embedding_cols
            filtered_cols = sample_cols(
                feature_ctgs.embedding_cols, ratio=0.20, logger=logger
            )
        case 'vectorized':
            # filtered_cols = feature_ctgs.tfidf_vectorized_cols
            filtered_cols = sample_cols(
                feature_ctgs.tfidf_vectorized_cols, ratio=0.20, logger=logger
            )
        
        case _:
            filtered_cols = feature_ctgs.structural_cols


    for col in filtered_cols:
        data = df[col].dropna()
        if data.empty: continue
    
        # skew = round(stats.skew(df[col].dropna()), 3)
        # kurt = round(stats.kurtosis(df[col].dropna()), 3)
        # mean = round(df[col].mean(), 3)
        # std = round(df[col].std(), 3)
        # 1. Existing Stats
        skew = round(stats.skew(data), 3)
        kurt = round(stats.kurtosis(data), 3)
        mean = data.mean()
        std = data.std()

        # summary.append((col, skew, kurt, mean, std))
        # 2. New Stability & Sparsity Metrics
        variance = round(data.var(), 3)
        
        # Coefficient of Variation (CV = Std / Mean)
        # Use a conditional to avoid division by zero
        cv = round(std / mean, 3) if mean != 0 else np.nan
        
        # Sparsity: Percentage of values that are exactly 0
        pct_zeros = round((data == 0).sum() / len(data) * 100, 2)

        summary.append((
            col, skew, kurt, round(mean, 3), round(std, 3), 
            variance, cv, pct_zeros
        ))

    # Log as single line
    logger.log_result(
        f"Numeric distributions summary (col_name, skew, kurt, mean, std): {summary}"
    )

    if experiment_dir:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        # Convert summary list to a DataFrame
        # Create the DataFrame with the new columns
        stats_df = pd.DataFrame(
            summary, 
            columns=['Feature', 'Skew', 'Kurt', 'Mean', 'Std', 'Var', 'CV', 'Pct_Zeros']
        )

        md_path = experiment_dir / "stats_summary.md"
        stats_df.to_markdown(md_path, index=False)

    cols = filtered_cols
    n_cols = 5
    n_rows = (len(cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        # sns.histplot is more reliable for overlays than displot in a loop
        sns.histplot(
            data=df,
            x=col,
            kde=True,
            ax=axes[i],
            bins=50,
            stat="density",  # Crucial: aligns the hist scale with the KDE scale
            color="#a8dadc",  # Light blue-grey for bars
            edgecolor="white",
            line_kws={
                "color": "#e63946",  # Sharp crimson for the KDE line
                "linewidth": 3,  # Thicker line for better visibility
                "label": "KDE",
            },
        )
        # axes[i].set_title(f"Dist: {col}")
        axes[i].set_title(f"Feature: {col}", fontsize=10, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Density")

    # Remove empty subplots if columns aren't a perfect multiple of 4
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_plt_image(experiment_dir=experiment_dir)

    # fig.suptitle("Numeric Feature Distributions with KDE (Density Normalized)", fontsize=16)
    fig.suptitle("Feature Distributions (Bars: Density | Line: KDE)", fontsize=16)
    plt.show()


def plot_embedding_distributions(
    df: pd.DataFrame, feature_ctgs: FeatureCategories, logger: MyLogger, experiment_dir: Path = None,

):
    logger.log_check("Checking Embedding Feature Distributions...")

    # def sample_embeddings(
    #     embedding_cols, logger: MyLogger, ratio=0.05, random_state=RANDOM_STATE
    # ):
    #     n_total = len(embedding_cols)
    #     n_sample = max(1, int(n_total * ratio))
    #     logger.log_result(
    #         f"Sampling {n_sample} out of {n_total} embedding columns for plotting."
    #     )

    #     rng = np.random.default_rng(random_state)
    #     return rng.choice(embedding_cols, size=n_sample, replace=False).tolist()

    # sampled_emb_cols = sample_embeddings(
    #     feature_ctgs.embedding_cols, ratio=0.05, logger=logger
    # )

    df[sampled_emb_cols].hist(bins=50, figsize=(20, 15))
    plt.suptitle(
        f"Embedding Distributions (Random {len(sampled_emb_cols)}/{len(feature_ctgs.embedding_cols)})"
    )

    if experiment_dir:
        experiment_dir.mkdir(parents=True, exist_ok=True)

        save_plt_image(experiment_dir=experiment_dir)

    plt.show()


def plot_file_types_distributions(df: pd.DataFrame, logger: MyLogger):
    logger.log_check("Checking File Extension Distribution...")

    # Compute extensions
    df["ext"] = df["filepath"].str.split(".").str[-1]

    # Counts
    ext_counts = df["ext"].value_counts()
    logger.log_result(f"Extension counts: {ext_counts.to_dict()}")

    # Proportions
    ext_props = (ext_counts / len(df)).round(4)
    logger.log_result(f"Extension proportions: {ext_props.to_dict()}")
    # Plot
    plt.figure(figsize=(8, 4))
    sns.barplot(x=ext_counts.index, y=ext_counts.values)
    plt.title("File Extension Distribution")
    plt.ylabel("Count")
    plt.show()


# %%
