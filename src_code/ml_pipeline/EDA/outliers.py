
import math
from typing import Iterable
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from notebooks.logging_config import MyLogger


def plot_boxplots_with_outliers(df: pd.DataFrame, features: Iterable[str], logger: MyLogger, cols: int = 4):
    logger.log_check("Boxplot outlier analysis (feature, outlier%, bounds, mean, std, min, max)...")

    rows = math.ceil(len(features) / cols)
    plt.figure(figsize=(cols * 5, rows * 3.5))

    for i, feature in enumerate(features, 1):
        series = df[feature].dropna()

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_mask = (series < lower) | (series > upper)
        outlier_ratio = float(outlier_mask.mean())
        n_outliers = int(outlier_mask.sum())

        # Additional stats
        mean = float(series.mean())
        std = float(series.std())
        min_val = float(series.min())
        max_val = float(series.max())

        # ---- LOG ENTRY (1 line per feature) ----
        logger.log_result(
            f"{feature}: outliers={outlier_ratio:.2%} ({n_outliers} rows), "
            f"bounds=({lower:.3f}, {upper:.3f}), "
            f"min={min_val:.3f}, max={max_val:.3f}, "
            f"mean={mean:.3f}, std={std:.3f}"
        )

        # ---- PLOT ----
        plt.subplot(rows, cols, i)
        sns.boxplot(x=series, linewidth=1)
        plt.title(f"{feature}\nOutliers: {outlier_ratio:.2%}")

    plt.tight_layout()
    plt.show()
