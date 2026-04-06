from argparse import ArgumentParser
import json
from pathlib import Path
import random
from typing import Iterable, List, Literal

from pydantic import TypeAdapter

from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERED_DATA_DIR, EXPERIMENT_DIR, LOG_DIR
from src_code.ml_pipeline.data_utils import load_df
from src_code.ml_pipeline.experimenting.types import Experiment
from src_code.versioning import VersionedFileManager

from .scripts.EDA import get_parser as eda_parser
from .scripts.engineer import get_parser as engineer_parser
from .scripts.tune import get_parser as tune_parser
from .scripts.preprocess import get_parser as prep_parser
from .scripts.train import get_parser as train_parser
from .scripts.evaluate import get_parser as eval_parser

from .scripts.EDA import perform_EDA
from .scripts.engineer import early_preprocess
from .scripts.tune import tune_hyperparams
from .scripts.preprocess import transform_df
from .scripts.train import train
from .scripts.evaluate import evaluate


MLPhase = Literal["eda", "engineer", "tune", "preprocess", "train", "eval"]

DEF_SCRIPT_LOGGER = MyLogger(
    label="ML_PIPELINE",
    section_name="ML PIPELINE LOGGER",
    file_log_path=LOG_DIR / "ml_pipeline_log.log",
)

EXPERIMENT_FILE = EXPERIMENT_DIR / "experiments.json"


# Use TypeAdapter to handle lists of Pydantic models
experiment_list_adapter = TypeAdapter(List[Experiment])

def load_experiments(file_path: Path):
    experiments: List[Experiment] = []

    # 1. Try to load existing experiments
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                # Use TypeAdapter to parse the list of dicts into List[Experiment]
                experiments = experiment_list_adapter.validate_python(json.load(f))
        except (json.JSONDecodeError, ValueError):
            # Handle empty or corrupted files
            experiments = []

    return experiments

def get_or_create_experiment(experiments: List[Experiment], logger: MyLogger, **new_exp_kwargs) -> Experiment:
    # experiments: List[Experiment] = []

    # # 1. Try to load existing experiments
    # if file_path.exists():
    #     try:
    #         with open(file_path, "r") as f:
    #             # Use TypeAdapter to parse the list of dicts into List[Experiment]
    #             experiments = experiment_list_adapter.validate_python(json.load(f))
    #     except (json.JSONDecodeError, ValueError):
    #         # Handle empty or corrupted files
    #         experiments = []
    # experiments = load_experiments(file_path=file_path)

    # 2. Search for the first unfinished experiment
    for exp in experiments:
        if not exp.is_finished:
            logger.log_result(f"📋 Resuming unfinished experiment: {exp.experiment_id}")
            return exp

    # 3. Create new if none found or file didn't exist
    logger.log_result("🚀 No unfinished experiments found. Creating a new one...")
    new_exp = Experiment(
        # experiment_id=Experiment.generate_id(),
        **new_exp_kwargs
    )
    
    # Save it immediately so the ID is reserved in the file
    # save_experiments(file_path, experiments + [new_exp])
    experiments.append(new_exp)
    return new_exp

def save_experiments(file_path: Path, experiments: List[Experiment]):
    with open(file_path, "w") as f:
        # model_dump handles Path and datetime conversion to JSON-safe types
        json_data = [exp.model_dump(mode='json') for exp in experiments]
        json.dump(json_data, f, indent=4)
    


if __name__ == "__main__":
    logger = DEF_SCRIPT_LOGGER
    parser = ArgumentParser(description="ML Pipeline")

    sub = parser.add_subparsers(dest="phase", required=True)

    sub.add_parser("eda", parents=[eda_parser()])
    sub.add_parser("engineer", parents=[engineer_parser()])
    sub.add_parser("tune", parents=[tune_parser()])
    sub.add_parser("preprocess", parents=[prep_parser()])
    sub.add_parser("train", parents=[train_parser()])
    sub.add_parser("eval", parents=[eval_parser()])

    run = sub.add_parser("run")
    run.add_argument(
        "phases", nargs="+", choices=["eda", "preprocess", "tune", "train", "evaluate"]
    )
    run.add_argument("--dataset", required=True)

    args = parser.parse_args()

    # args.phase = (MLPhase)(args.phase)
    experiment_id = random.randint(1, 1000)
    # experiments: List[Experiment] = []
    experiments = load_experiments(file_path=EXPERIMENT_FILE)
    curr_experiment = get_or_create_experiment(experiments=experiments, logger=logger)

    if args.phase == "eda":
        curr_experiment.eda_results = perform_EDA(
            subset=args.subset,
            experiment_id=experiment_id,
            max_rows=args.max_rows,
            intersect_with_processed=args.intersect_with_processed,
        )

        # curr_experiment.eda_results = results
    if args.phase == "engineer":
        curr_experiment.preprocessing_results = early_preprocess(
            subset=args.subset,
            max_rows=args.max_rows,
            experiment_id=experiment_id,
            # logger=logger,
        )
    elif args.phase == "tune":
        # early_preprocessed_df_versioner = VersionedFileManager(
        #     file_path=ENGINEERED_DATA_DIR / "train_engineered", logger=logger
        # )
        # early_preprocessed_df = load_df(
        #     df_file_path=early_preprocessed_df_versioner.current_newest, logger=logger
        # )

        curr_experiment.tuning_results = tune_hyperparams(
            model_type=args.model,
            # logger=logger,
            experiment_id=experiment_id,
            max_rows=args.max_rows,
            n_cores=args.workers,
            core_mode=args.core_mode,
            reserve_cores=args.reserve_cores,
        )
        # curr_experiment.tuning_results = results
    # elif args.phase == "preprocess":
    #     transform_df(
    #         subset=args.subset,
    #         max_rows=args.max_rows,
    #         # script_logger=logger,
    #         experiment_id=experiment_id,
    #     )
    elif args.phase == "train":
        curr_experiment.training_results.append(train(
            model_type=args.model,
            load_tuned=args.load_tuned,
            skip_pfi=args.skip_pfi,
            top_k=args.top_k,
            experiment_id=experiment_id,
        ))
    elif args.phase == "eval":
        curr_experiment.eval_results = evaluate(models=args.models, experiment_id=experiment_id)
    else:
        raise ValueError(f"Invalid value: {args.phase=}")
        # for 
    
    save_experiments(file_path=EXPERIMENT_FILE, experiments=experiments)
    
    
