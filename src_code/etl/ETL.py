import argparse
import os
import threading
from typing import List, Literal
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from functools import partial
from tqdm import tqdm
import time
from git import Repo
import logging
import pyarrow.feather as feather

from notebooks.logging_config import MyLogger
from src_code.etl.utils import get_repo_instance
from src_code.ml_pipeline.resources import CoreConfig
from src_code.versioning import VersionedFileManager

from .repos import (
    BUG_INDUCING_COMMITS,
    download_missing_repo_clones,
    get_missing_repo_clones,
    get_registered_repos,
    is_registered,
    load_bug_inducing_comms,
)
from .extraction import extract_commit_features
from .features.developer_social import (
    pre_calculate_author_metrics,
)
from .features.historical_temporal import (
    calc_recent_churn_from_df,
)


from ..config import *


# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.DEBUG)

# Create a console handler and set the level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

stop_event = threading.Event()


class StopProcessing(Exception):
    """Custom exception to stop processing in worker threads."""

    pass


# ---------------------------------------------------------------------
# Commit classifier (now also extracts features)
# ---------------------------------------------------------------------
# def classify_and_extract(row, bug_set: set):
#     # 💡 CRITICAL: Check stop event before starting expensive work (e.g., git clone)
#     if stop_event.is_set():
#         # Raise our custom exception to cleanly exit the thread
#         raise StopProcessing(f"Stop event set. Skipping {row.repo}/{row.commit}")

#     repo = row.repo
#     commit_hash = row.commit
#     filepath = row.filepath

#     label = 1 if commit_hash in bug_set else 0

#     # 2. Feature Extraction
#     extraction_start = time.time()
#     features = extract_commit_features(Repo(PYTHON_LIBS_DIR / repo), commit_hash)
#     extraction_end = time.time()
#     logger.info(
#         f"[TIMING] Feature extraction for {repo}/{commit_hash} took {extraction_end - extraction_start:.2f} seconds."
#     )
#     # Merge results, critically including 'repo' and 'commit' for safe merging later
#     result = {
#         "repo": repo,
#         "commit": commit_hash,
#         "filepath": filepath,
#         "label": label,
#         **features,
#     }

#     return result

def normalize_grouped_lines(value):
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        return []

    normalized = []
    for inner in value:
        if isinstance(inner, np.ndarray):
            inner = inner.tolist()
        elif inner is None:
            inner = []
        elif not isinstance(inner, list):
            inner = []
        normalized.append(inner)

    return normalized


def has_buggy_lines(grouped_lines) -> bool:
    rows = normalize_grouped_lines(grouped_lines)
    return any(len(inner) > 0 for inner in rows)

# def classify_and_extract(row, bug_set: set):
#     if stop_event.is_set():
#         raise StopProcessing(f"Stop event set. Skipping {row.repo}/{row.commit}")

#     repo = row.repo
#     commit_hash = row.commit
#     label = row.label

#     # label = 1 if commit_hash in bug_set else 0

#     extraction_start = time.time()
#     features = extract_commit_features(Repo(PYTHON_LIBS_DIR / repo), commit_hash, logger)
#     extraction_end = time.time()

#     logger.info(
#         f"[TIMING] Feature extraction for {repo}/{commit_hash} took {extraction_end - extraction_start:.2f} seconds."
#     )

#     result = {
#         "repo": repo,
#         "commit": commit_hash,
#         "label": label,
#         **features,
#     }

#     return result

def classify_and_extract(row):
    if stop_event.is_set():
        raise StopProcessing(f"Stop event set. Skipping {row.repo}/{row.commit}")

    repo = row.repo
    commit_hash = row.commit
    label = row.label

    extraction_start = time.time()
    features = extract_commit_features(Repo(PYTHON_LIBS_DIR / repo), commit_hash, logger)
    extraction_end = time.time()

    logger.info(
        f"[TIMING] Feature extraction for {repo}/{commit_hash} took {extraction_end - extraction_start:.2f} seconds."
    )

    if features is None:
        features = {}

    result = {
        "repo": repo,
        "commit": commit_hash,
        "label": label,
        "lines": row.lines,              # optional, but useful to keep
        "has_bug": row.has_bug,          # optional
        **features,
    }

    return result

def atomic_feather_save(df: pd.DataFrame, out_file: Path):
    tmp_file = out_file.with_suffix(".tmp")
    feather.write_feather(df, tmp_file)
    os.replace(tmp_file, out_file)  # atomic on POSIX


def can_create_file(path: Path) -> bool:
    parent = path.parent

    # Parent must exist
    if not parent.exists():
        return False

    # Path cannot be an existing directory
    if path.exists() and path.is_dir():
        return False

    # Check write permission to parent directory
    return os.access(parent, os.W_OK)


# def _save_to_file(
#     input_df: pd.DataFrame,
#     results_list: list,
#     existing_out_file: Path,
#     out_file: Path,
#     append: bool,
# ):
#     """

#     Args:
#         input_df (pd.DataFrame): DataFrame containing the original input data.
#         results_list (list): List of dictionaries containing the results from processing.
#         existing_out_file (Path): Path to the existing output file.
#         out_file (Path): Path to the output file.
#         append (bool): Whether to append to the existing file or not.
#     """

#     # Convert results back to a DataFrame
#     results_df = pd.DataFrame(results_list)

#     # 💡 CRITICAL CHANGE: Use merge for a SAFE partial save.
#     # This aligns the features/labels using the common keys ('repo', 'commit'),
#     # ensuring data integrity even if rows were skipped or completed out of order.
#     # df_merged = input_df.merge(
#     #     results_df,
#     #     on=["repo", "commit", "filepath"],
#     #     how="inner",  # Only keep rows that were successfully processed
#     # )
#     df_merged = (
#         input_df[["repo", "commit"]]
#         .drop_duplicates()
#         .merge(results_df, on=["repo", "commit"], how="inner")
#     )

#     df_merged["recent_churn"] = calc_recent_churn_from_df(df_merged, window_days=30)

#     # Drop embeddings if they are not part of results_df
#     for col in ["code_embed", "msg_embed"]:
#         if col not in results_df.columns and col in df_merged.columns:
#             df_merged = df_merged.drop(columns=[col], errors="ignore")

#     # Convert any remaining np.ndarray to list, keep existing lists
#     for col in ["code_embed", "msg_embed"]:
#         if col in df_merged.columns:
#             df_merged[col] = df_merged[col].apply(
#                 lambda x: x.tolist() if isinstance(x, np.ndarray) else x
#             )
#             df_merged[col] = df_merged[col].apply(
#                 lambda x: x if isinstance(x, list) else []
#             )
#             df_merged[col] = df_merged[col].astype(object)

#     # Guard against empty results
#     if len(results_df) == 0:
#         logger.warning("No results to save; skipping file write.")
#         return

#     if append and existing_out_file.exists():
#         logger.info(
#             f"Append=True and {existing_out_file} exists → loading existing data..."
#         )

#         existing_df = pd.read_feather(existing_out_file)

#         key_cols = ["repo", "commit", "filepath"]

#         # 🔥 Compare ONLY the newly processed rows
#         before = len(df_merged)
#         df_new = df_merged.merge(
#             existing_df[key_cols], on=key_cols, how="left", indicator=True
#         )

#         df_new = df_new[df_new["_merge"] == "left_only"].drop(columns=["_merge"])
#         after = len(df_new)

#         logger.info(f"Skipping existing rows: {before - after} duplicates removed.")
#         logger.info(f"Appending {after} new rows.")

#         # Append new rows only
#         final_df = pd.concat([existing_df, df_new], ignore_index=True)
#         final_df.drop_duplicates(subset=key_cols, keep="last", inplace=True)

#         # atomic_feather_save(final_df, out_file)
#         final_df.to_feather(out_file)

#         return

#     logger.info(f"Saving {len(df_merged)} rows to {out_file}")
#     df_merged.to_feather(out_file)
#     logger.info("Done")
def _save_to_file(
    input_df: pd.DataFrame,
    results_list: list,
    existing_out_file: Path,
    out_file: Path,
    append: bool,
):
    results_df = pd.DataFrame(results_list)

    if len(results_df) == 0:
        logger.warning("No results to save; skipping file write.")
        return

    key_cols = ["repo", "commit"]

    # Keep only newly extracted columns from results_df, plus keys
    results_df = results_df[
        [c for c in results_df.columns if c in key_cols or c not in input_df.columns]
    ]

    df_merged = input_df.merge(results_df, on=key_cols, how="inner")

    df_merged["recent_churn"] = calc_recent_churn_from_df(df_merged, window_days=30)

    # Drop embeddings if they are not part of results_df
    for col in ["code_embed", "msg_embed"]:
        if col not in results_df.columns and col in df_merged.columns:
            df_merged = df_merged.drop(columns=[col], errors="ignore")

    for col in ["code_embed", "msg_embed"]:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
            df_merged[col] = df_merged[col].apply(
                lambda x: x if isinstance(x, list) else []
            )
            df_merged[col] = df_merged[col].astype(object)

    if append and existing_out_file.exists():
        logger.info(
            f"Append=True and {existing_out_file} exists → loading existing data..."
        )

        existing_df = pd.read_feather(existing_out_file)

        before = len(df_merged)
        df_new = df_merged.merge(
            existing_df[key_cols], on=key_cols, how="left", indicator=True
        )

        df_new = df_new[df_new["_merge"] == "left_only"].drop(columns=["_merge"])
        after = len(df_new)

        logger.info(f"Skipping existing rows: {before - after} duplicates removed.")
        logger.info(f"Appending {after} new rows.")

        final_df = pd.concat([existing_df, df_new], ignore_index=True)
        final_df.drop_duplicates(subset=key_cols, keep="last", inplace=True)

        final_df.to_feather(out_file)
        return

    logger.info(f"Saving {len(df_merged)} rows to {out_file}")
    df_merged.to_feather(out_file)
    logger.info("Done")


# ---------------------------------------------------------------------
# Main transformation logic
# ---------------------------------------------------------------------
def transform(
    repos_filter: List[str] = None,
    reserved_cores: int = 8,
    skip_existing: bool = False,
    save_after: int = None,
    subset: Literal["train", "test", "validate"] = "train",
):
    """
    Args:
        repos_filter (List[str], optional): D
        workers (int, optional): _description_. Defaults to 8.
        skip_existing (bool, optional): _description_. Defaults to False.
        save_after (int, optional): _description_. Defaults to None.
        subset (Literal[&#39;train&#39;, &#39;test&#39;, &#39;validate&#39;], optional): _description_. Defaults to 'train'.

    Raises:
        e: _description_
    """

    # --- DataFrame File Path Constants ---
# --- Logging Path Constants ---
    myLogger = MyLogger(label="ETL", section_name="ETL_script", file_log_path=LOG_DIR / "etl.log")
    # raw input DataFrame path
    input_df_path: Path = ETL_PATH_MAPPINGS[subset]["input"]
    # newest output DataFrame path processed by this script
    existing_versioner = VersionedFileManager(
        file_path=INTERIM_DATA_DIR / f"{subset}_labeled_features_partial.feather",
        logger=myLogger,
    )
    existing_output_df_path: Path = existing_versioner.current_newest
    # existing_output_df_path: Path = ETL_PATH_MAPPINGS[subset]["current_newest"]
    # next output DataFrame path to write new results
    # next_output_df_path: Path = ETL_PATH_MAPPINGS[subset]["next_output"]
    next_output_df_path = existing_versioner.next_base_output

    

    logger.info(f"Newest out file: {existing_output_df_path}")
    logger.info(f"Next out file: {next_output_df_path}")

    logger.info(f"[LOAD] {input_df_path}")
    logger.info(f"Using {reserved_cores} reserved cores.")
    logger.info(f"Repos filter: {repos_filter}")

    # --- Loading Raw Data in Feather format ---

    input_df = pd.read_feather(input_df_path)
    existing_output_df = pd.read_feather(existing_output_df_path)

    # --- Filtering to Specific Repos only ---

    if repos_filter and len(repos_filter) > 0:
        logger.info(f"[FILTER] Limiting to repos: {repos_filter}")
        input_df = input_df[input_df["repo"].isin(repos_filter)]
        logger.info(f"Dataset size after filtering: {len(input_df)}")
    else:
        logger.info("No repository filter applied. Processing ALL repositories.")

    # --- Skipping already processed rows ---

    if skip_existing and existing_output_df_path.exists():
        logger.info(
            f"skip_existing=True and {existing_output_df_path} exists → loading existing data..."
        )

        key_cols = ["repo", "commit"]

        # before = len(input_df)

        input_df = input_df.merge(
            existing_output_df[key_cols].drop_duplicates(),
            on=key_cols,
            how="left",
            indicator=True
        )
        before = len(input_df)


        input_df = input_df[input_df["_merge"] == "left_only"].drop(columns=["_merge"])

        after = len(input_df)

        logger.info(f"Skipping already existing rows: {before - after} duplicates removed.")
        logger.info(f"{after} rows remain to process.")
        
    core_mode = CoreConfig(reserve_cores=reserved_cores, mode="all")

    logger.info(f"Dataset size: {len(input_df)}")
    logger.info(f"[PROCESS] Starting feature extraction with {core_mode.n_jobs} workers...")

    # CRITICAL ADDITION: Pre-calculate the expensive metrics
    logger.info("[PRECALC] Calculating author experience and recent activity...")
    # NOTE: You MUST ensure your input DF has an 'author' column (email) and 'datetime' column.
    input_df = pre_calculate_author_metrics(input_df, get_repo_func=get_repo_instance)
    logger.info("[PRECALC] Finished calculating author metrics.")

    if len(input_df) == 0:
        logger.warning("Dataset is empty after filtering. Exiting.")
        return

    # We need the 'repo' and 'commit' columns present in the input for merging later
    # rows_to_process = list(
    #     input_df[["repo", "commit", "filepath"]].itertuples(index=False)
    # )
    # commit_input_df = (
    # input_df[["repo", "commit"]]
    #     .drop_duplicates()
    #     .reset_index(drop=True)
    # )
    # commit_input_df = (
    #     input_df
    #     .groupby(["repo", "commit"], as_index=False)
    #     .first()
    # )
    # commit_input_df = (
    #     input_df
    #     .groupby(["repo", "commit"], as_index=False)
    #     .agg({
    #         "lines": list,
    #         "content": lambda x: "\n".join(map(str, x)),
    #         "filepath": list,   # optional, useful for debugging
    #     })
    # )
    agg_dict = {
        "lines": list,
        "content": lambda x: "\n".join(map(str, x)),
    }

    for col in ["filepath", "author_email", "canonical_datetime", "author_exp_pre", "author_recent_activity_pre"]:
        if col in input_df.columns:
            agg_dict[col] = list if col == "filepath" else "first"

    commit_input_df = (
        input_df
        .groupby(["repo", "commit"], as_index=False)
        .agg(agg_dict)
    )


    commit_input_df["has_bug"] = commit_input_df["lines"].apply(has_buggy_lines)
    commit_input_df["label"] = commit_input_df["has_bug"].astype(int)

    rows_to_process = list(commit_input_df.itertuples(index=False))
    results_list = []

    classify_func = partial(classify_and_extract)

    # --- Parallel Execution Block with Ctrl+C Handling ---
    logger.info("Press Ctrl+C to stop processing and save partial results.")

    with ThreadPoolExecutor(max_workers=core_mode.n_jobs) as executor:
        # Map original row tuples to futures
        futures = {
            # executor.submit(classify_func, row, BUG_INDUCING_COMMITS[row.repo]): row
            executor.submit(classify_func, row): row
            for row in rows_to_process
        }

        pbar = tqdm(
            total=len(rows_to_process), desc="Extracting Features", unit="commit"
        )

        try:
            # Use as_completed and tqdm for progress and interruptibility
            # for future in pbar:
            for future in as_completed(futures):
                pbar.update(1)

                if stop_event.is_set():
                    # If stop_event was set in a signal handler, break the loop
                    logger.info("Stop event set, breaking result collection loop.")
                    break

                try:
                    result = future.result()
                    results_list.append(result)

                    if save_after and len(results_list) >= save_after:
                        logger.info(f"Saving {save_after} processed rows!")
                        _save_to_file(
                            input_df=commit_input_df,
                            results_list=results_list,
                            existing_out_file=existing_output_df_path,
                            append=skip_existing,
                            out_file=next_output_df_path,
                        )
                        results_list.clear()
                        existing_output_df_path = next_output_df_path

                except StopProcessing as e:
                    logger.info(f"Worker stopped: {e}")

                except CancelledError:
                    pass
                except Exception as e:
                    row = futures.get(future, "UNKNOWN")
                    logger.error(
                        f"\nA task failed for commit {row.repo}/{row.commit}: {e}"
                    )
                    raise e

        except KeyboardInterrupt:
            # Main thread receives Ctrl+C
            logger.warning(
                "\nCtrl+C received! Setting stop_event and cancelling futures..."
            )
            stop_event.set()
            pbar.close()

            # Cancel all remaining futures
            for future in futures:
                future.cancel()

    # --- Post-Processing and Saving ---

    if not results_list:

        logger.warning(
            "No results generated (or interrupted immediately). Exiting without saving."
        )
        return

    _save_to_file(
        input_df=commit_input_df,
        results_list=results_list,
        existing_out_file=existing_output_df_path,
        append=skip_existing,
        out_file=next_output_df_path,
    )

    logger.info(f"Commit-level rows: {len(commit_input_df)}")
    logger.info(f"Positive label rate: {commit_input_df['label'].mean():.4f}")

    sample = commit_input_df[["repo", "commit", "lines", "label"]].head(5)
    logger.info(f"Grouped commit sample:\n{sample}")

    bad = commit_input_df[
        (commit_input_df["label"] == 0) & (commit_input_df["lines"].apply(has_buggy_lines))
    ]
    logger.info(f"Inconsistent clean labels after grouping: {len(bad)}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform JIT dataset with labels.")
    parser.add_argument(
        "--repos",
        nargs="*",
        help="List of repositories to include (default: all). Example: --repos airflow pandas",
    )

    parser.add_argument(
        "--subset",
        choices=["train", "test", "validate"],  # This is the key part
        required=False,  # Recommend making it required
        default="train",  # Optional: Set a default value
        help="The data subset to process. Must be one of: train, test, or validate.",
    )

    # parser.add_argument(
    #     "--workers",
    #     type=int,
    #     default=8,
    #     help="Number of worker threads for parallel classification",
    # )
    parser.add_argument(
        "--reserve-cores",
        type=int,
        default=4
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Append to existing output file if it exists, skipping already processed commits.",
    )
    parser.add_argument(
        "--save-after",
        type=int,
        default=1000,
        help="Stores after N processed rows. Defaults to no limit.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="If true, all registered repos will be extracted.",
        required=False,
        default=False,
    )

    parser.add_argument(
        "--missing-repos",
        action="store_true",
        help="If true, all registered repos with missing clones will be listed.",
        required=False,
        default=False,
    )

    args = parser.parse_args()

    if args.missing_repos:
        missing_repos = get_missing_repo_clones()
        logger.info(f"Missing Repo Clones: {missing_repos}")

        if missing_repos:
            while True:
                resp = input("Download them now? (y/n): ")

                if resp == "y":
                    download_missing_repo_clones()
                elif resp != "n":
                    continue

                break

    if args.all or args.repos:
        logger.warning("No repo specified. Defaulting to registered repos.")
    else:
        for repo in args.repos:
            if not is_registered(repo):
                raise ValueError(
                    f"Invalid --repo value: {repo}. This repository is not registered."
                )

    if args.save_after and args.save_after < 1:
        raise ValueError(
            f"Invalid --save-after value: {args.save_after}. Must be a positive number (>0)."
        )

    if args.all or args.repos:
        load_bug_inducing_comms()

        transform(
            subset=args.subset,
            repos_filter=args.repos if not args.all else get_registered_repos(),
            reserved_cores=args.reserve_cores,
            skip_existing=args.skip_existing,
            save_after=args.save_after,
        )
