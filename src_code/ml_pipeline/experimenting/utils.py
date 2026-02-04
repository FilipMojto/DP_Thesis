
from pathlib import Path
from notebooks.logging_config import MyLogger
from matplotlib import pyplot as plt


def log_experiment_id(logger: MyLogger, experiment_id: int):
    logger.log_check(
        f"Experiment ID: {experiment_id}"
        if experiment_id
        else "No Experiment ID provided."
    )

def save_plt_image(experiment_dir: Path):
    if experiment_dir:
        save_file = experiment_dir / "precision_recall_curves.png"
        plt.savefig(save_file)
        print(f"Saved PR grid to {save_file}")