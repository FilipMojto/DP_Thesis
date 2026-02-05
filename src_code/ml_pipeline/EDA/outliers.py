
import math
from pathlib import Path
from typing import Iterable
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.EDA.plots import grid_paginator
from src_code.ml_pipeline.EDA.utils import NumFeatureSets
from src_code.utils.utils import is_binary


# def plot_boxplots_with_outliers(df: pd.DataFrame, features: Iterable[str], logger: MyLogger, cols: int = 4, experiment_dir: Path = None):
#     logger.log_check("Boxplot outlier analysis (feature, outlier%, bounds, mean, std, min, max)...")

#     rows = math.ceil(len(features) / cols)
#     plt.figure(figsize=(cols * 5, rows * 3.5))

#     gen = grid_paginator(features, 'structural', experiment_dir, n_cols=1, rows_per_page=5, preset='a4-portrait')


#     for i, feature in enumerate(features, 1):
#         series = df[feature].dropna()

#         Q1 = series.quantile(0.25)
#         Q3 = series.quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR

#         outlier_mask = (series < lower) | (series > upper)
#         outlier_ratio = float(outlier_mask.mean())
#         n_outliers = int(outlier_mask.sum())

#         # Additional stats
#         mean = float(series.mean())
#         std = float(series.std())
#         min_val = float(series.min())
#         max_val = float(series.max())

#         # ---- LOG ENTRY (1 line per feature) ----
#         logger.log_result(
#             f"{feature}: outliers={outlier_ratio:.2%} ({n_outliers} rows), "
#             f"bounds=({lower:.3f}, {upper:.3f}), "
#             f"min={min_val:.3f}, max={max_val:.3f}, "
#             f"mean={mean:.3f}, std={std:.3f}"
#         )

#         # ---- PLOT ----
#         plt.subplot(rows, cols, i)
#         sns.boxplot(x=series, linewidth=1)
#         plt.title(f"{feature}\nOutliers: {outlier_ratio:.2%}")

#     plt.tight_layout()
#     plt.show()

def plot_boxplots_with_outliers(
    df: pd.DataFrame, 
    feature_sets: NumFeatureSets, 
    logger: MyLogger, 
    experiment_dir: Path = None,
    drop_binaries: bool = True,
):
    logger.log_check("Boxplot outlier analysis with paginated output...")

    features = feature_sets.numeric_cols + feature_sets.engineered_cols

    # if drop_binaries:
    #     logger.log_check("Dropping binaries...")

    #     for feature in features:
    #         if is_binary(df=df, col_name=feature):
    #             df = df.drop(feature, axis=1)
    #             logger.log_result(f"Dropped binary column: {feature}")
    #             features.remove(feature)
    if drop_binaries:
        logger.log_check("Dropping binaries...")

        binaries = [f for f in features if is_binary(df=df, col_name=f)]

        if binaries:
            df = df.drop(columns=binaries)
            features = [f for f in features if f not in binaries]

            for f in binaries:
                logger.log_result(f"Dropped binary column: {f}")
        



    # 1. Initialize the paginator
    # n_cols=1 and rows_per_page=5 fits 5 stacked boxplots on an A4 page nicely
    gen = grid_paginator(
        features=features,
        col_name='structural_boxplots', 
        experiment_dir=experiment_dir, 
        n_cols=1, 
        rows_per_page=5, 
        preset='a4-portrait'
    )

    # 2. Iterate directly over the generator
    for ax, feature in gen:
        series = df[feature].dropna()

        # Calculation Logic
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_mask = (series < lower) | (series > upper)
        outlier_ratio = float(outlier_mask.mean())
        
        # Log results
        logger.log_result(
            f"{feature}: outliers={outlier_ratio:.2%}, "
            f"bounds=({lower:.2f}, {upper:.2f}), mean={series.mean():.2f}"
        )

        # 3. Plot onto the specific 'ax' provided by the paginator
        sns.boxplot(x=series, ax=ax, linewidth=1, fliersize=3)
        ax.set_title(f"{feature} (Outliers: {outlier_ratio:.2%})")
        ax.set_xlabel("") # Clean up labels for a grid look

    # Note: plt.show() or plt.savefig() is handled inside grid_paginator's loop