import pandas as pd

from notebooks.logging_config import MyLogger
from src_code.config import INTERIM_DATA_DIR, LOG_DIR
from src_code.ml_pipeline.data_utils import load_df
from src_code.versioning import VersionedFileManager


def analyze_bug_inducing_lines_distribution(input_df: pd.DataFrame, logger: MyLogger):
    input_df["has_bug"] = input_df["lines"].apply(lambda x: len(x) > 0)

    grouped = input_df.groupby(["repo", "commit"])

    analysis = grouped.apply(
        lambda g: pd.Series(
            {
                "any_bug": g["has_bug"].any(),
                "first_bug": g["has_bug"].iloc[0],  # first file
            }
        )
    )

    # commits that actually contain bugs
    buggy_commits = analysis[analysis["any_bug"]]

    # how many of those have bug in first file
    correct_first = buggy_commits["first_bug"].sum()
    total_buggy = len(buggy_commits)

    percentage = (correct_first / total_buggy) * 100

    logger.log_result(
        f"{correct_first}/{total_buggy} ({percentage:.2f}%) buggy commits have bug in first file"
    )


if __name__ == "__main__":
    logger = MyLogger(
        file_log_path=LOG_DIR / "relabeling_analysis.log",
        label="relabeling_analysis",
        section_name="Analysis",
    )
    subset = "train"

    input_df_versioner = VersionedFileManager(
        file_path=INTERIM_DATA_DIR / f"{subset}_labeled_features_partial.feather",
        logger=logger,
    )

    input_df = load_df(df_file_path=input_df_versioner.current_newest, logger=logger)
    input_df.info()
    analyze_bug_inducing_lines_distribution(input_df, logger)
