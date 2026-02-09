from argparse import ArgumentParser
import random
from typing import Literal

from notebooks.logging_config import MyLogger
from src_code.config import ENGINEERED_DATA_DIR, LOG_DIR
from src_code.ml_pipeline.data_utils import load_df
from src_code.versioning import VersionedFileManager

from .scripts.engineer import get_parser as engineer_parser
from .scripts.tune import get_parser as tune_parser
from .scripts.preprocess import get_parser as prep_parser

from .scripts.engineer import early_preprocess
from .scripts.tune import tune_hyperparams
from .scripts.preprocess import transform_df


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

    # sub.add_parser("eda", parents=[eda_parser()])
    # sub.add_parser("preprocess", parents=[preprocess_parser()])
    sub.add_parser("engineer", parents=[engineer_parser()])
    sub.add_parser("tune", parents=[tune_parser()])
    sub.add_parser("preprocess", parents=[prep_parser()])

    args = parser.parse_args()

    # args.phase = (MLPhase)(args.phase)
    experiment_id = random.randint(1, 1000)

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
            n_workers=args.workers,
        )
    elif args.phase == "preprocess":
        transform_df(
            subset=args.subset,
            max_rows=args.max_rows,
            # script_logger=logger,
            experiment_id=experiment_id,
        )
