from argparse import ArgumentParser
import random
from typing import Literal

from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERED_DATA_DIR, LOG_DIR
from src_code.ml_pipeline.data_utils import load_df
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

    if args.phase == "eda":
        perform_EDA(
            subset=args.subset,
            experiment_id=experiment_id,
            max_rows=args.max_rows,
            intersect_with_processed=args.intersect_with_processed,
        )

    if args.phase == "engineer":
        early_preprocess(
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

        tune_hyperparams(
            model_type=args.model,
            # logger=logger,
            experiment_id=experiment_id,
            max_rows=args.max_rows,
            n_cores=args.workers,
            core_mode=args.core_mode,
            reserve_cores=args.reserve_cores,
        )
    elif args.phase == "preprocess":
        transform_df(
            subset=args.subset,
            max_rows=args.max_rows,
            # script_logger=logger,
            experiment_id=experiment_id,
        )
    elif args.phase == "train":
        train(
            model_type=args.model,
            load_tuned=args.load_tuned,
            skip_pfi=args.skip_pfi,
            top_k=args.top_k,
            experiment_id=experiment_id,
        )
    elif args.phase == "eval":
        evaluate(models=args.models, experiment_id=experiment_id)
