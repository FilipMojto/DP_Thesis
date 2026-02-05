import math
from pathlib import Path
from typing import Dict, Iterable, List, Literal

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from notebooks.constants import EMBEDDINGS, ENGINEERED_FEATURES, LINE_TOKEN_FEATURES
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis

from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.EDA.utils import FeatureCategories
from src_code.ml_pipeline.EDA.utils import FeatureCategories
from src_code.ml_pipeline.experimenting.utils import save_df_as_md, save_plt_as_image
from src_code.ml_pipeline.scripts.train import RANDOM_STATE


# def plot_label_distribution(df: pd.DataFrame, logger: MyLogger):
#     logger.log_check("Checking Label Distribution...")

#     # Plot
#     sns.countplot(x="label", data=df)
#     plt.title("Bug Label Distribution")
#     plt.show()

#     # Raw counts
#     label_counts = df["label"].value_counts()
#     logger.log_result(
#         f"Label counts: {[(int(k), int(v)) for k, v in label_counts.items()]}"
#     )

#     # Proportions
#     label_props = df["label"].value_counts(normalize=True)
#     logger.log_result(
#         f"Label proportions: {[(int(k), round(float(v), 4)) for k, v in label_props.items()]}"
#     )

def grid_paginator(
    features: Iterable[str],
    col_name: str,
    experiment_dir: Path,
    n_cols: int = 1,
    rows_per_page: int = 3,
    preset: str = "a4-portrait"
):
    """Manages the creation of multi-page PDF grids for LaTeX."""
    setup_latex_style(preset)
    
    total_features = len(features)
    features_per_page = n_cols * rows_per_page
    num_pages = math.ceil(total_features / features_per_page)

    for page in range(num_pages):
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, total_features)
        page_features = features[start_idx:end_idx]

        fig, axes = plt.subplots(rows_per_page, n_cols)
        # Ensure axes is always an iterable array even for 1x1
        axes_list = np.atleast_1d(axes).flatten()

        for i, feature in enumerate(page_features):
            yield axes_list[i], feature

        # Cleanup unused axes on the last page
        for j in range(len(page_features), len(axes_list)):
            fig.delaxes(axes_list[j])

        plt.tight_layout()
        if experiment_dir:
            suffix = f"_p{page+1}" if num_pages > 1 else ""
            save_path = experiment_dir / f"{col_name}_dist{suffix}.pdf"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()
            

# def plot_feature_distribution(df: pd.DataFrame, feature: str, logger: MyLogger, rotation: int = 0, exp_dir: Path = None):
#     logger.log_check(f"Checking commits per {feature}...")

#     repo_counts = df[feature].value_counts()

#     # Plot
#     plt.figure(figsize=(10, 5))
#     sns.barplot(x=repo_counts.index, y=repo_counts.values)
#     plt.title(f"Commits per {feature}")
#     plt.ylabel("Number of Commits")
#     plt.xticks(rotation=rotation)

#     if exp_dir:
#         exp_dir.mkdir(parents=True, exist_ok=True)
#         save_plt_as_image(experiment_dir=exp_dir, file_name=f"{feature}_distribution")
#     else:
#         plt.show()

#     # Raw counts (one-line log)
#     logger.log_result(
#         f"{feature} commit counts: {[(repo, int(count)) for repo, count in repo_counts.items()]}"
#     )

#     # Proportions (one-line log)
#     repo_props = repo_counts / len(df)
#     logger.log_result(
#         f"{feature} commit proportions: {[(repo, round(float(prop), 4)) for repo, prop in repo_props.items()]}"
#     )
def plot_categorical_comparison(
    dfs: Dict[str, pd.DataFrame], 
    feature: str,
    logger: MyLogger,
    experiment_dir: Path = None,
    rows_per_page: int = 3
):
    """Compares categorical distributions across multiple dataframes."""
    
    # We use our generator to handle the layout
    gen = grid_paginator([feature], feature, experiment_dir, n_cols=1, rows_per_page=rows_per_page)
    
    for ax, feature in gen:
        # Combine counts from all DFs for comparison
        plot_data = []
        for name, df in dfs.items():
            counts = df[feature].value_counts(normalize=True).reset_index()
            counts.columns = [feature, 'proportion']
            counts['Source'] = name
            plot_data.append(counts)
        
        comparison_df = pd.concat(plot_data)

        sns.barplot(
            data=comparison_df, 
            x=feature, 
            y='proportion', 
            hue='Source', 
            ax=ax,
            palette="muted"
        )
        ax.set_title(f"Comparison: {feature}", fontweight="bold")
        ax.set_ylabel("Proportion of Commits")
        ax.legend(title="Dataset")

    logger.log_result(f"Categorical comparison for {feature} completed.")


def setup_latex_style(preset="a4-portrait"):
    """Sets global matplotlib params for LaTeX consistency."""
    if preset == "a4-portrait":
        # Text width is approx 6.3 inches.
        # We set figure width to match exactly to avoid LaTeX scaling.
        plt.rcParams.update(
            {
                "figure.figsize": (6.3, 9.0),  # Almost full A4 height
                "font.size": 9,
                "axes.labelsize": 9,
                "axes.titlesize": 10,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "lines.linewidth": 1.5,
            }
        )
    elif preset == "monitor":
        plt.rcParams.update({"figure.figsize": (16, 9), "font.size": 12})


def plot_num_feature_distributions(
    df: pd.DataFrame,
    logger: MyLogger,
    feature_ctgs: FeatureCategories,
    col_type: Literal[
        "structural", "engineered", "embedding", "vectorized"
    ] = "structural",
    experiment_dir: Path = None,
    drop_cols: Iterable[str] = None,
    preset="a4-portrait",
    rows_per_page=6,
):
    
    def sample_cols(cols, logger: MyLogger, ratio=0.05, random_state=RANDOM_STATE):
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
        case "structural":
            filtered_cols = [
                feature
                for feature in feature_ctgs.structural_cols
                if feature not in ENGINEERED_FEATURES
            ]
        case "engineered":
            filtered_cols = [
                feature
                for feature in feature_ctgs.structural_cols
                if feature in ENGINEERED_FEATURES
            ]
        case "embedding":
            # filtered_cols = feature_ctgs.embedding_cols
            filtered_cols = sample_cols(
                feature_ctgs.embedding_cols, ratio=0.20, logger=logger
            )
        case "vectorized":
            # filtered_cols = feature_ctgs.tfidf_vectorized_cols
            filtered_cols = sample_cols(
                feature_ctgs.tfidf_vectorized_cols, ratio=0.20, logger=logger
            )

        case _:
            filtered_cols = feature_ctgs.structural_cols

    if drop_cols:
        filtered_cols = [col for col in filtered_cols if col not in drop_cols]

    for col in filtered_cols:
        data = df[col].dropna()
        if data.empty:
            continue

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

        summary.append(
            (col, skew, kurt, round(mean, 3), round(std, 3), variance, cv, pct_zeros)
        )

    # Log as single line
    logger.log_result(
        f"Numeric distributions summary (col_name, skew, kurt, mean, std): {summary}"
    )

    stats_df = pd.DataFrame(
        summary,
        columns=[
            "Feature",
            "Skew",
            "Kurt",
            "Mean",
            "Std",
            "Var",
            "CV",
            "Pct_Zeros",
        ],
    )
    
    if experiment_dir:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        # Convert summary list to a DataFrame
        # Create the DataFrame with the new columns

        # md_path = experiment_dir / "stats_summary.md"
        # stats_df.to_markdown(md_path, index=False)
        save_df_as_md(df=stats_df, path=experiment_dir / f"{col_type}_stats_summary.md")
    
    gen = grid_paginator(filtered_cols, col_type, experiment_dir, n_cols=1, rows_per_page=rows_per_page, preset=preset)
    
    for ax, feature in gen:
        sns.histplot(
            data=df, x=feature, kde=True, ax=ax,
            stat="density", color="#a8dadc", edgecolor="white",
            line_kws={"color": "#e63946", "linewidth": 2}
        )
        ax.set_title(f"Distribution: {feature}", fontweight="bold")
        ax.set_xlabel("")

    # setup_latex_style(preset)

    # # Calculate how many pages we need
    # total_features = len(filtered_cols)
    # features_per_page = features_per_page
    # num_pages = math.ceil(total_features / features_per_page)

    # for page in range(num_pages):
    #     start_idx = page * features_per_page
    #     end_idx = min(start_idx + features_per_page, total_features)
    #     page_features = filtered_cols[start_idx:end_idx]

    #     # Grid layout for this page (e.g., 3 rows x 2 columns)
    #     # n_cols = 2
    #     # n_rows = math.ceil(len(page_features) / n_cols)
    #     # Inside your plot function, change the grid calculation:
    #     n_cols = 1  # Forced to 1 for side-by-side comparison
    #     n_rows = features_per_page  # Usually 3

    #     # Create figure using the preset figsize
    #     fig, axes = plt.subplots(n_rows, n_cols)
    #     axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    #     for i, col in enumerate(page_features):
    #         sns.histplot(
    #             data=df,
    #             x=col,
    #             kde=True,
    #             ax=axes[i],
    #             stat="density",
    #             color="#a8dadc",
    #             edgecolor="white",
    #             line_kws={"color": "#e63946", "linewidth": 2},
    #         )
    #         axes[i].set_title(f"Feature: {col}", fontweight="bold")
    #         axes[i].set_xlabel("")  # Keep clean for LaTeX

    #     # Clean up empty slots
    #     for j in range(len(page_features), len(axes)):
    #         fig.delaxes(axes[j])

    #     plt.tight_layout()

    #     # Save each page separately
    #     suffix = f"_p{page+1}" if num_pages > 1 else ""

    #     if experiment_dir:
    #         save_path = experiment_dir / f"{col_type}_dist{suffix}.pdf"
    #         plt.savefig(save_path, bbox_inches="tight", dpi=300)
    #     else:
    #         plt.show()
    #     plt.close()
    
    return stats_df



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


def plot_commit_msg_len_distribution(df: pd.DataFrame, logger: MyLogger):
    logger.log_check("Checking Commit Message Length Distribution...")

    # --- Commit message lengths ---
    msg_stats = df["msg_len"].describe().to_dict()
    logger.log_result(f"msg_len stats: {msg_stats}")
    plt.figure(figsize=(8, 4))
    sns.histplot(df["msg_len"], bins=50)
    plt.title("Commit Message Length Distribution")
    plt.show()

    # # --- Empty content rows ---
    # empty_content = int((df['content'].str.len() == 0).sum())
    # logger.log_result(f"Empty content rows: {empty_content}")


def plot_line_discrepancy_distribution(df: pd.DataFrame, logger: MyLogger):
    # --- Line discrepancy analysis ---
    logger.log_check(
        "Checking line count discrepancy (content_lines - loc_added - loc_deleted)..."
    )

    line_discrepancy = (
        df["content"].str.count("\n") - df["loc_added"] - df["loc_deleted"]
    )
    disc_stats = line_discrepancy.describe().to_dict()

    # Convert numpy values to python floats
    disc_stats = {k: float(v) for k, v in disc_stats.items()}

    logger.log_result(f"line_discrepancy stats: {disc_stats}")

    # Optional plot (not logged)
    plt.figure(figsize=(8, 4))
    sns.histplot(line_discrepancy, bins=50)
    plt.title("Line Discrepancy Distribution")
    plt.show()


def plot_embedding_norm_distribution(df: MyLogger, logger: MyLogger):
    logger.log_check(
        "Checking Embedding Norm Distributions (NAME, SKEW, KURT, MEAN, STD)..."
    )

    df["code_embed_norm"] = df["code_embed"].apply(lambda x: np.linalg.norm(x))
    df["msg_embed_norm"] = df["msg_embed"].apply(lambda x: np.linalg.norm(x))

    plt.figure(figsize=(12, 5))
    sns.histplot(df["code_embed_norm"], bins=50, kde=True)
    plt.title("Code Embedding Norm Distribution")
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.histplot(df["msg_embed_norm"], bins=50, kde=True)
    plt.title("Message Embedding Norm Distribution")
    plt.show()

    for col in ["code_embed_norm", "msg_embed_norm"]:
        series = df[col].dropna()
        series_skew = skew(series)
        series_kurt = kurtosis(series)
        series_mean = series.mean()
        series_std = series.std()

    logger.log_result(
        f"{col}: skew={series_skew:.3f}, kurt={series_kurt:.3f}, mean={series_mean:.3f}, std={series_std:.3f}"
    )


def plot_feature_relationships_with_label(
    df: pd.DataFrame, cols: Iterable[str], logger: MyLogger
):
    logger.log_check(
        "Checking Feature Distributions by Label... (median, IQR, min, max, outlier_ratio)"
    )

    # structural_cols = feature_ctgs.structural_cols.copy()
    # if "except" in structural_cols:
    #     structural_cols.remove("except")

    # Logging metrics
    for feature in cols:
        for lbl in df["label"].unique():
            series = df[df["label"] == lbl][feature].dropna()
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            median = series.median()
            min_val, max_val = series.min(), series.max()
            outlier_ratio = (
                (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
            ).mean()
            logger.log_result(
                f"{feature} | label={lbl} | median={median:.2f}, IQR={IQR:.2f}, min={min_val:.2f}, max={max_val:.2f}, outlier_ratio={outlier_ratio:.2%}"
            )

    # Plotting
    n_features = len(cols)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    plt.figure(figsize=(20, 4 * n_rows))

    for i, col in enumerate(cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(x="label", y=col, data=df)
        plt.title(f"{col} by label")

    plt.tight_layout()
    plt.show()


def plot_keyword_distributions(
    df: pd.DataFrame, logger: MyLogger, features=LINE_TOKEN_FEATURES
):
    logger.log_check(
        "Checking Keyword Count Distributions... (median, IQR, min, max, zero_ratio)"
    )

    plt.figure(figsize=(15, 5))
    df[features].hist(bins=50, figsize=(15, 5))
    plt.suptitle("Keyword Counts")
    plt.show()

    for feature in features:
        series = df[feature].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        median = series.median()
        min_val, max_val = series.min(), series.max()
        zero_ratio = (series == 0).mean()

        logger.log_result(
            f"{feature} | median={median:.2f}, IQR={IQR:.2f}, min={min_val}, max={max_val}, zero_ratio={zero_ratio:.2%}"
        )


def plot_parwise_relationship(
    df: pd.DataFrame,
    feature_ctgs: FeatureCategories,
    logger: MyLogger,
    top_features: int = 5,
):
    # TOP_FEATURES = 5

    # Compute correlations with label
    corr_with_label = (
        df[feature_ctgs.all_numeric_cols() + ["label"]].corr()["label"].drop("label")
    )  # exclude label itself

    # Take top N correlated features
    selected_features = (
        corr_with_label.abs()
        .sort_values(ascending=False)
        .head(top_features)
        .index.to_list()
    )

    # Add label back for hue
    selected_features.append("label")

    # print(selected_features)

    # # selected_features = ['loc_added', 'loc_deleted', 'hunks_count', 'max_func_change', 'label']
    # sns.pairplot(df[selected_features], hue='label', corner=True)
    # plt.suptitle("Pairplot of Selected Features", y=1.02)
    # plt.show()

    # selected_features
    logger.log_check(
        "Checking Selected Feature Distributions and Correlations... (median, IQR, min, max, overlap)"
    )

    # Exclude label for stats
    features_only = selected_features[:-1]

    # Log per-feature stats + overlap coefficient
    for feature in features_only:
        series0 = df[df["label"] == 0][feature].dropna()
        series1 = df[df["label"] == 1][feature].dropna()

        # Basic stats on full feature
        median = df[feature].median()
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        min_val, max_val = df[feature].min(), df[feature].max()

        # Overlap between label groups
        bins = np.histogram_bin_edges(df[feature], bins="auto")
        hist0, _ = np.histogram(series0, bins=bins, density=True)
        hist1, _ = np.histogram(series1, bins=bins, density=True)
        overlap = np.sum(np.minimum(hist0, hist1)) * np.diff(bins)[0]

        logger.log_result(
            f"{feature} | median={median:.2f}, IQR={IQR:.2f}, min={min_val:.2f}, max={max_val:.2f}, overlap={overlap:.2f}"
        )

    # Log pairwise correlations among selected features (excluding label)
    for i, feat1 in enumerate(features_only):
        for feat2 in features_only[i + 1 :]:
            corr_val = df[[feat1, feat2]].corr().iloc[0, 1]
            logger.log_result(f"Correlation {feat1} & {feat2}: {corr_val:.2f}")

    # Pairplot for visualization
    sns.pairplot(df[selected_features], hue="label", corner=True)
    plt.suptitle("Pairplot of Selected Features", y=1.02)
    plt.show()


def plot_2D_embedding_separability(df: pd.DataFrame, logger: MyLogger, embeddings = EMBEDDINGS, sample_size: int = 2000):
    # sample_size = 2000  # safe for memory

    for emb_name in embeddings:
        logger.log_check(f"PCA of {emb_name}")

        # Stack embeddings into 2D array
        emb_matrix = np.vstack(df[emb_name].values)

        # PCA to 2D
        # pca = PCA(n_components=2)
        # emb_2d = pca.fit_transform(emb_matrix)
        pca = PCA(n_components=2)
        # Force the result to be a NumPy array immediately
        emb_2d = np.array(pca.fit_transform(emb_matrix))

        # Explained variance
        var_explained = pca.explained_variance_ratio_
        logger.log_result(f"{emb_name} - Explained variance by PC1: {var_explained[0]:.2%}, PC2: {var_explained[1]:.2%}")


        # 1. Split by labels (Ensure these are numpy arrays)
        labels = df['label'].values
        points_label0 = emb_2d[labels == 0]
        points_label1 = emb_2d[labels == 1]

        # 2. Fixed Sampling: Use iloc if it's a DF, or index directly if it's an array
        # To be safe, we force points_label0 to be a numpy array if it isn't already
        if hasattr(points_label0, 'values'):
            points_label0 = points_label0.values
        if hasattr(points_label1, 'values'):
            points_label1 = points_label1.values

        idx0 = np.random.choice(points_label0.shape[0], min(sample_size, points_label0.shape[0]), replace=False)
        idx1 = np.random.choice(points_label1.shape[0], min(sample_size, points_label1.shape[0]), replace=False)

        points_label0_sample = points_label0[idx0]
        points_label1_sample = points_label1[idx1]

        # Distances
        inter_label_dist = pairwise_distances(points_label0_sample, points_label1_sample).mean()
        intra_dist_label0 = pairwise_distances(points_label0_sample).mean()
        intra_dist_label1 = pairwise_distances(points_label1_sample).mean()

        logger.log_result(f"{emb_name} - Mean distance between label 0 and 1 in PCA 2D space: {inter_label_dist:.3f}")
        logger.log_result(f"{emb_name} - Mean intra-label distances: label 0: {intra_dist_label0:.3f}, label 1: {intra_dist_label1:.3f}")

        # Plot
        plt.figure(figsize=(10,7))
        plt.scatter(emb_2d[:,0], emb_2d[:,1], c=df['label'], cmap='coolwarm', alpha=0.5)
        plt.title(f"PCA of {emb_name} Colored by Label")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()