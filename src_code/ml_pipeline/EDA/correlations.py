

from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from notebooks.constants import TARGET
from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.EDA.plots import setup_latex_style


def select_top_correlated_features(
    corr: pd.DataFrame,
    target: str = TARGET,
    top_n: int | None = None,
    min_abs_corr: float | None = None,
):
    corr_target = corr[target].drop(target).dropna()

    # 🔹 Filter by minimum absolute correlation
    if min_abs_corr is not None:
        corr_target = corr_target[corr_target.abs() >= min_abs_corr]

    # 🔹 Sort by absolute correlation
    corr_sorted = corr_target.loc[
        corr_target.abs().sort_values(ascending=False).index
    ]

    # 🔹 Apply top-N
    if top_n is not None:
        corr_sorted = corr_sorted.head(top_n)

    return corr_sorted.index.tolist(), corr_sorted


# def plot_corr_matrix(
#     corr_matrix: pd.DataFrame,
#     label: str,
#     logger: MyLogger,
#     target: str = TARGET,
#     top_n: int = None,
#     min_abs_corr: float | None = None,
#     experiment_dir: Path = None
# ):
#     logger.log_check("Computing correlation matrix...")

#     if top_n:
#         top_features, corr_vals = select_top_correlated_features(
#             corr=corr_matrix, target=target, top_n=top_n, min_abs_corr=min_abs_corr
#         )
#         selected_cols = top_features + [target]
#         # corr_matrix = corr_matrix[selected_cols]
#         corr_matrix = corr_matrix.loc[selected_cols, selected_cols]
#         logger.log_result(
#             f"Selected {len(top_features)} features "
#             f"(top_n={top_n}, min_abs_corr={min_abs_corr})",
#             print_to_console=True,
#         )

#     # corr_matrix = df[structural_cols + ['label']].corr()
#     plt.figure(figsize=(22, 20))

#     sns.heatmap(
#         corr_matrix,
#         annot=True,
#         fmt=".2f",
#         cmap="coolwarm",
#         center=0,
#         square=True,
#     )
#     plt.title("Correlation Matrix")
#     plt.tight_layout()

#     if experiment_dir:
#         experiment_dir.mkdir(parents=True, exist_ok=True)
#         setup_latex_style("a4-portrait")
#         # suffix = f"_p{page+1}" if num_pages > 1 else ""
#         suffix = ""
#         save_path = experiment_dir / f"{label}_heatmap.pdf"
#         plt.savefig(save_path, bbox_inches="tight", dpi=300)
#     else:
#         plt.show()
def plot_corr_matrix(
    corr_matrix: pd.DataFrame,
    label: str,
    logger: MyLogger,
    target: str = TARGET,
    top_n: int = 15,  # Keep this manageable for a single page
    min_abs_corr: float | None = None,
    experiment_dir: Path = None
):
    logger.log_check("Computing correlation matrix...")

    # 1. Feature Selection
    if top_n:
        top_features, _ = select_top_correlated_features(
            corr=corr_matrix, target=target, top_n=top_n, min_abs_corr=min_abs_corr
        )
        selected_cols = top_features + [target]
        corr_matrix = corr_matrix.loc[selected_cols, selected_cols]

    # 2. Apply LaTeX Style BEFORE creating the figure
    setup_latex_style("a4-portrait")
    
    # 3. Calculate dynamic height based on number of features
    # A4 width is ~6.3. We want square cells, so height approx = width
    fig_width = 6.3 
    fig_height = min(fig_width * (len(selected_cols) / 10), 9.0) # Scale height but cap at page limit
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 4. Professional Heatmap Styling
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r", # Red-Blue is more standard for academic papers
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8}, # Make colorbar slightly smaller
        annot_kws={"size": 7},    # Smaller font for numbers to prevent overlap
        ax=ax
    )
    
    ax.set_title(f"Correlation Matrix: {label}", pad=20)
    
    # 5. Save with tight bounding box
    if experiment_dir:
        save_path = experiment_dir / f"{label}_heatmap.pdf"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        # Use bbox_inches="tight" to ensure no labels are cut off
        plt.savefig(save_path, format='pdf', bbox_inches="tight")
        logger.log_result(f"Saved thesis-ready heatmap to {save_path}")
    else:
        plt.show()



def compute_correlations_with_target(
    corr_matrix: pd.DataFrame,
    logger: MyLogger,
    target: str = TARGET,
    top_n: int | None = None,
    min_abs_corr: float | None = None,
) -> dict:
    """
    Computes correlations of features with the target and returns
    a dictionary {feature_name: correlation_value}, sorted by absolute correlation.
    
    Parameters:
        corr_matrix : pd.DataFrame
            Correlation matrix containing all features and target.
        logger : MyLogger
            Logger for progress messages.
        target : str
            Name of the target column.
        top_n : int | None
            If set, return only the top N features by absolute correlation.
        min_abs_corr : float | None
            If set, include only features with absolute correlation >= min_abs_corr.
    
    Returns:
        dict : {feature_name: correlation_with_target}
    """
    logger.log_check("Computing correlations with label...")

    # Extract correlations with target, drop the target itself
    corr_with_target = corr_matrix[target].drop(target).dropna()

    # Filter by minimum absolute correlation if specified
    if min_abs_corr is not None:
        corr_with_target = corr_with_target[corr_with_target.abs() >= min_abs_corr]

    # Sort by absolute correlation (strongest first)
    corr_sorted = corr_with_target.reindex(corr_with_target.abs().sort_values(ascending=False).index)

    # Apply top_n if specified
    if top_n is not None:
        corr_sorted = corr_sorted.head(top_n)

    # Convert to dictionary
    corr_dict = corr_sorted.to_dict()
    logger.log_result(f"Best correlations: {corr_dict}")

    return corr_dict