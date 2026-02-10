
import argparse
from pathlib import Path
from pathlib import Path
from typing import Any, Callable, Dict, Iterable
import pandas as pd
from matplotlib import pyplot as plt

from notebooks.logging_config import MyLogger
from src_code.ml_pipeline.experimenting.types import ARG_RESOLVER, ARG_VALIDATOR, Config


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



def get_experiment_dir(experiment_id: int, target_dir: Path) -> Path:
    path = Path(f"{target_dir}/experiment_{experiment_id}")
    path.mkdir(parents=True, exist_ok=True)
    return path


class MyParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def resolve_args(self, args: Config, resolvers: Dict[str, ARG_RESOLVER]):
    #     # args = self.parse_args()
    #     return {name: resolver(args) for name, resolver in resolvers.items()}

    # def validate(
    #     self,
    #     args: Config,
    #     validators: Iterable[ARG_VALIDATOR],
    # ) -> None:
    #     for validator in validators:
    #         validator(args)
    pass


# MyParser(desc)