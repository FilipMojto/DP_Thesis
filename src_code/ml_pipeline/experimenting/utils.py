
from pathlib import Path

import pandas as pd
from notebooks.logging_config import MyLogger
from matplotlib import pyplot as plt


def log_experiment_id(logger: MyLogger, experiment_id: int):
    logger.log_check(
        f"Experiment ID: {experiment_id}"
        if experiment_id
        else "No Experiment ID provided."
    )


def save_plt_as_image(experiment_dir: Path, file_name: str):
    if experiment_dir:
        save_file = experiment_dir / f"{file_name}.pdf"
        plt.savefig(save_file, bbox_inches="tight")
        print(f"Saved PR grid to {save_file}")


def save_df_as_md(df: pd.DataFrame, path: Path):
    df.to_markdown(path, index=False)
