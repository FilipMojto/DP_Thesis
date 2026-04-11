import argparse
from typing import get_args

import pandas as pd

from notebooks.constants import TARGET
from notebooks.logging_config import MyLogger
from src_code.config import EXTENDED_DATA_DIR, INTERIM_DATA_DIR, LOG_DIR, RELABELED_DATA_DIR, SubsetType
from src_code.ml_pipeline.data_utils import load_df, save_df
from src_code.versioning import VersionedFileManager


def verify_input_df(df, logger: MyLogger):
    # check if the df has the expected columns
    expected_columns = {"commit", TARGET}
    if not expected_columns.issubset(set(df.columns)):
        err_msg = f"Input df is missing expected columns. Expected at least: {expected_columns}. Got: {set(df.columns)}"
        logger.logger.error(err_msg)
        raise ValueError(err_msg)

    if df[TARGET].isnull().any():
        err_msg = "Input df contains null values in the target column."
        logger.logger.error(err_msg)
        raise ValueError(err_msg)
    
    # check if the target column is binary
    if not set(df[TARGET].unique()).issubset({0, 1}):
        err_msg = f"Input df target column contains non-binary values. Got: {set(df[TARGET].unique())}"
        logger.logger.error(err_msg)
        raise ValueError(err_msg)
    
    logger.log_result(f"Input df has the expected columns: {expected_columns}")


def check_inconsistent_labels(df, logger: MyLogger):
    has_bug = df["lines"].apply(lambda rows: any(len(x) > 0 for x in rows))

    inconsistent_1 = df[(df[TARGET] == 0) & (has_bug)]
    inconsistent_2 = df[(df[TARGET] == 1) & (~has_bug)]

    if not inconsistent_1.empty or not inconsistent_2.empty:
        err_msg = (
            f"Inconsistent labels detected: "
            f"{len(inconsistent_1)} false negatives, "
            f"{len(inconsistent_2)} false positives"
        )
        logger.logger.error(err_msg)
        raise ValueError(err_msg)
    else:
        logger.log_result("No inconsistent labels detected between target and lines.")



# def relabel_target(row, logger: MyLogger):
#     # Placeholder for the actual relabeling logic
#     # dataset contains git commits
#     # lines attribute contains line numbers of the buggy code segment, if any. If lines is empty, then the commit is not buggy. If lines is not empty, then the commit is buggy.    
    
#     # df[TARGET] = 1 - df[TARGET]  # Invert binary labels (0 becomes 1, and 1 becomes 0)
#     # logger.log_result("Successfully relabeled the target column.")

#     if row[TARGET] == 0 and row["lines"]:  # If the commit is labeled as non-buggy but has associated lines, relabel as buggy
#         row[TARGET] = 1
#         logger.log_result(f"Relabeled a non-buggy commit {row['commit']} with associated lines as buggy.")
#     elif row[TARGET] == 1 and not row["lines"]:  # If the commit is labeled as buggy but has no associated lines, relabel as non-buggy
#         row[TARGET] = 0
#         logger.log_result(f"Relabeled a buggy commit {row['commit']} with no associated lines as non-buggy.")
    
#     logger.log_result("Successfully relabeled the target column.")
#     return row


def check_label_diff(df_before: pd.DataFrame, df_after: pd.DataFrame, logger: MyLogger):
    keys = ["repo", "commit"]

    # Merge datasets on keys
    merged = df_before[keys + [TARGET]].merge(
        df_after[keys + [TARGET]],
        on=keys,
        how="inner",
        suffixes=("_before", "_after")
    )

    # Compute differences
    label_changes = (merged[f"{TARGET}_before"] != merged[f"{TARGET}_after"]).sum()
    total_rows = len(merged)

    logger.log_result(
        f"Relabeled {label_changes} out of {total_rows} aligned rows "
        f"({(label_changes / total_rows) * 100:.2f}%)"
    )

    logger.log_result(f"df_before size: {len(df_before)}")
    logger.log_result(f"df_after size: {len(df_after)}")
    logger.log_result(f"merged size: {len(merged)}")

    before_keys = set(zip(df_before["repo"], df_before["commit"]))
    after_keys = set(zip(df_after["repo"], df_after["commit"]))

    missing_in_after = before_keys - after_keys
    missing_in_before = after_keys - before_keys

    logger.log_result(f"Missing in df_after: {len(missing_in_after)}")
    logger.log_result(f"Missing in df_before: {len(missing_in_before)}")


def check_labeled_data(df: pd.DataFrame, logger: MyLogger):
    # print random sample of relabeled data
    sample_size = min(5, len(df))
    sample = df.sample(sample_size, random_state=42)
    # modify lines column to show only few starting and ending line numbers for better readability
    # sample["lines"] = sample["lines"].apply(lambda x: f"{x[:3]}...{x[-3:]}" if isinstance(x, list) and len(x) > 6 else x)
    logger.log_result(f"Random sample of relabeled data:\n{sample}")

    print("Lines: ")
    for _, row in sample.iterrows():
        print(f"  {row['commit']}: {row['lines']}")


def relabel_data(input_df: pd.DataFrame, logger: MyLogger) -> pd.DataFrame:
    df_size_before = len(input_df)
    # input_df = input_df.drop_duplicates(subset=keys)
    
    commit_df = (
        input_df
        .groupby(["repo", "commit"])
        .agg({
            "lines": list,  # keep all lists
        })
        .reset_index()
    )

    commit_df["has_bug"] = commit_df["lines"].apply(
        lambda rows: any(len(x) > 0 for x in rows)
    )

    df_size_after = len(commit_df)

    # commit_df["label"] = commit_df["lines"].astype(int)
    commit_df["label"] = commit_df["has_bug"].astype(int)

    logger.log_result(
        f"Dropped {df_size_before - df_size_after} ({(df_size_before - df_size_after) / df_size_before}%) duplicates!"
    )

    return commit_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relabel target based on different labelling logic.")
    parser.add_argument(
        "--subset",
        type=str,
        choices=get_args(SubsetType),
        required=True,
        help="Subset to relabel (train, test, validate)"
    )
    logger = MyLogger(label="relabel_data", section_name="relabel_data", file_log_path=LOG_DIR / "relabel_data.log")

    args = parser.parse_args()
    subset: SubsetType = args.subset

    input_df_versioner = VersionedFileManager(
        file_path=INTERIM_DATA_DIR / f"{subset}_labeled_features_partial.feather", logger=logger
    )
    original_df = load_df(df_file_path=EXTENDED_DATA_DIR / f"{subset}_extended_v2.feather", logger=logger)
    output_df_versioner = VersionedFileManager(
        file_path=RELABELED_DATA_DIR / f"{subset}_relabelled.feather", logger=logger
    )

    # relabeled_df = relabel_data(input_df=input_df, logger=logger)
    

    input_df = load_df(df_file_path=input_df_versioner.current_newest, logger=logger)
    relabeled_df = relabel_data(input_df=input_df, logger=logger)

    verify_input_df(input_df, logger)
    check_inconsistent_labels(relabeled_df, logger)
    check_label_diff(df_before=load_df(df_file_path=RELABELED_DATA_DIR / f"{subset}_original_labeled.feather", logger=logger), df_after=relabeled_df, logger=logger)
    check_labeled_data(relabeled_df, logger)
    # relabeled_df = df.apply(lambda row: relabel_target(row, logger), axis=1)

    # save_df(df=relabeled_df, df_file_path=output_df_versioner.next_base_output, logger=logger)

    save_df(df=relabeled_df, df_file_path=output_df_versioner.next_base_output, logger=logger)

