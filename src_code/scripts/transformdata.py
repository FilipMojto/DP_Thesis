import argparse
import os
import tempfile
import threading
from typing import List
import numpy as np
import yaml
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from functools import lru_cache, partial
from tqdm import tqdm
import time
from git import Repo
import logging
import pyarrow.feather as feather

from src_code.preprocessing.repos import (
    BUG_INDUCING_COMMITS,
    is_registered,
    load_bug_inducing_comms,
)
from src_code.preprocessing.extraction import extract_commit_features
from src_code.preprocessing.features.developer_social import (
    pre_calculate_author_metrics,
)
from src_code.preprocessing.features.historical_temporal import (
    calc_recent_churn_from_df,
)
from src_code.utils.feature_extractor import FeatureExtractor


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

# # ---------------------------------------------------------------------
# # Load per-repo YAML files lazily and cache them
# # ---------------------------------------------------------------------
# @lru_cache(maxsize=None)
# def load_bug_inducing_for_repo(repo_name: str):
#     """Load bug-inducing commits for a given repo based on filename."""

#     yaml_files = list(BUG_INDUCING_DIR.glob(f"{repo_name}.*"))

#     if not yaml_files:
#         print(f"[WARN] No YAML found for repo: {repo_name}")
#         return set()

#     file_path = yaml_files[0]  # only one expected

#     with open(file_path, "r") as f:
#         data = yaml.safe_load(f) or {}
#         inducing_set = set()

#         for _, inducing_list in data.items():
#             inducing_set.update(inducing_list or [])

#     return inducing_set


# # Assuming you have a dictionary mapping repo names to URLs
# REPO_URL_MAP = {"pandas": "https://github.com/pandas-dev/pandas.git"}
# # --- END CONFIG PLACEHOLDERS ---

# BUG_INDUCING_COMMITS = {}

# for repo in REPO_URL_MAP.keys():
#     BUG_INDUCING_COMMITS[repo] = load_bug_inducing_for_repo(repo)

stop_event = threading.Event()


class StopProcessing(Exception):
    """Custom exception to stop processing in worker threads."""

    pass


# ---------------------------------------------------------------------
# Commit classifier (now also extracts features)
# ---------------------------------------------------------------------
def classify_and_extract(row, bug_set: set):
    # 💡 CRITICAL: Check stop event before starting expensive work (e.g., git clone)
    if stop_event.is_set():
        # Raise our custom exception to cleanly exit the thread
        raise StopProcessing(f"Stop event set. Skipping {row.repo}/{row.commit}")

    # print(f"[PROCESS] Repo: {row.repo}, Commit: {row.commit}")
    repo = row.repo
    commit_hash = row.commit
    filepath = row.filepath
    # bug_inducing_file = BUG_INDUCING_DIR / repo

    # 1. Classification
    # bug_set = load_bug_inducing_for_repo(repo)
    # print(repo, "bug set size:", len(bug_set))

    label = 1 if commit_hash in bug_set else 0
    # git_repo = extractor.get_repo(PYTHON_LIBS_DIR / repo) # Assuming this method exists on FeatureExtractor

    # 2. Feature Extraction
    # features = extractor.extract_features(repo, commit_hash)
    extraction_start = time.time()
    features = extract_commit_features(Repo(PYTHON_LIBS_DIR / repo), commit_hash)
    extraction_end = time.time()
    logger.info(
        f"[TIMING] Feature extraction for {repo}/{commit_hash} took {extraction_end - extraction_start:.2f} seconds."
    )
    # Merge results, critically including 'repo' and 'commit' for safe merging later
    result = {
        "repo": repo,
        "commit": commit_hash,
        "filepath": filepath,
        "label": label,
        **features,
    }
    # print(f"Processed commit {commit_hash} in repo {repo}: label={label}")

    # with open(bug_inducing_file)

    return result


def get_repo_instance(repo_name: str) -> Repo:
    """Helper function to load the local Git Repo object."""
    # Assuming PYTHON_LIBS_DIR points to where repos are cloned
    return Repo(PYTHON_LIBS_DIR / repo_name)


# def atomic_feather_save(df, out_file: Path):
#     """
#     Safely write a feather file atomically by writing to a temporary
#     file and moving it into place.
#     """
#     out_file = Path(out_file)
#     tmp_dir = out_file.parent

#     # Create temporary file in the same directory (important)
#     with tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir, suffix=".feather") as tmp:
#         tmp_path = Path(tmp.name)

#     try:
#         # Write feather content into the temporary file

#         ft.write_feather(df, tmp_path)

#         # Force flush to disk
#         os.sync()

#         # Atomically replace the target
#         tmp_path.replace(out_file)

#     except Exception as e:
#         # Ensure temp does not stick around
#         if tmp_path.exists():
#             tmp_path.unlink()
#         raise e


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


def _save_to_file(
    input_df: pd.DataFrame,
    results_list: list,
    existing_out_file: Path,
    out_file: Path,
    append: bool,
    # copy_out_file: bool,
):
    
    # Convert results back to a DataFrame
    results_df = pd.DataFrame(results_list)
    
    # if copy_out_file:
    #     out_file = existing_out_file.with_name(
    #         out_file.stem + "_copy" + out_file.suffix
    #     )

    # out_file = (
    #     existing_out_file.with_name(existing_out_file.stem + "_copy" + existing_out_file.suffix)
    #     if copy_out_file
    #     else existing_out_file
    # )

    # 💡 CRITICAL CHANGE: Use merge for a SAFE partial save.
    # This aligns the features/labels using the common keys ('repo', 'commit'),
    # ensuring data integrity even if rows were skipped or completed out of order.
    df_merged = input_df.merge(
        results_df,
        on=["repo", "commit", "filepath"],
        how="inner",  # Only keep rows that were successfully processed
    )

    df_merged["recent_churn"] = calc_recent_churn_from_df(df_merged, window_days=30)
    
    # for col in ['code_embed', 'msg_embed']:
    #     df_merged[col] = df_merged[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    #     df_merged[col] = df_merged[col].astype(object)

    #     # Before merge, ensure input_df columns won't collide
    # for col in ['code_embed', 'msg_embed']:
    #     if col not in results_df.columns and col in df_merged.columns:
    #         df_merged = df_merged.drop(columns=[col], errors='ignore')

    # # Ensure embeddings are object type
    # for col in ['code_embed', 'msg_embed']:
    #     if col in df_merged.columns:
    #         df_merged[col] = df_merged[col].apply(lambda x: x if isinstance(x, list) else [])
    #         df_merged[col] = df_merged[col].astype(object)

    # 1. What Python types exist in the column?
    # print("Type counts (code_embed):")
    # print(df_merged['code_embed'].map(lambda x: type(x)).value_counts(dropna=False))

    # print("\nType counts (msg_embed):")
    # print(df_merged['msg_embed'].map(lambda x: type(x)).value_counts(dropna=False))

    # 2. Sample representations (first 10)
    # print("\nSamples (repr) for code_embed:")
    for i, v in enumerate(df_merged['code_embed'].iloc[:10]):
        print(i, type(v), repr(v)[:200])

    # print("\nCheck 'isinstance(np.ndarray)' for first 20 rows:")
    # print(df_merged['code_embed'].apply(lambda x: isinstance(x, np.ndarray)).head(20).to_list())

    # 3. Column dtype and pandas internal array repr
    # print("\nColumn dtypes and pandas array representation:")
    # print(df_merged[['code_embed', 'msg_embed']].dtypes)
    # print(df_merged['code_embed']._values)   # may show NumpyExtensionArray or Arrow type

    # Drop embeddings if they are not part of results_df
    for col in ['code_embed', 'msg_embed']:
        if col not in results_df.columns and col in df_merged.columns:
            df_merged = df_merged.drop(columns=[col], errors='ignore')

    # Convert any remaining np.ndarray to list, keep existing lists
    for col in ['code_embed', 'msg_embed']:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            df_merged[col] = df_merged[col].apply(lambda x: x if isinstance(x, list) else [])
            df_merged[col] = df_merged[col].astype(object)

    # print("Type counts (code_embed):")
    # print(df_merged['code_embed'].map(lambda x: type(x)).value_counts(dropna=False))

    # print("\nType counts (msg_embed):")
    # print(df_merged['msg_embed'].map(lambda x: type(x)).value_counts(dropna=False))

    # 2. Sample representations (first 10)
    # print("\nSamples (repr) for code_embed:")
    for i, v in enumerate(df_merged['code_embed'].iloc[:10]):
        print(i, type(v), repr(v)[:200])

    # print("\nCheck 'isinstance(np.ndarray)' for first 20 rows:")
    # print(df_merged['code_embed'].apply(lambda x: isinstance(x, np.ndarray)).head(20).to_list())

    # 3. Column dtype and pandas internal array repr
    # print("\nColumn dtypes and pandas array representation:")
    # print(df_merged[['code_embed', 'msg_embed']].dtypes)
    # print(df_merged['code_embed']._values)   # may show NumpyExtensionArray or Arrow type

        
    # print(type(df_merged.loc[0, 'code_embed']))
    # len_check = len(df_merged.loc[0, 'code_embed'])

    # print(type(df_merged.loc[0, 'msg_embed']))
    # len_check = len(df_merged.loc[0, 'msg_embed'])

    # coltypes_check = df_merged.dtypes

    # something_cde = df_merged['code_embed'].apply(lambda x: isinstance(x, list)).value_counts()
    # sth_msg = df_merged['msg_embed'].apply(lambda x: isinstance(x, list)).value_counts()

    # empty_code_embed = (df_merged['code_embed'].apply(len) == 0).sum()
    # empty_ms_embed = (df_merged['msg_embed'].apply(len) == 0).sum()

    # Guard against empty results
    if len(results_df) == 0:
        logger.warning("No results to save; skipping file write.")
        return

    if append and existing_out_file.exists():
        logger.info(f"Append=True and {existing_out_file} exists → loading existing data...")

        existing_df = pd.read_feather(existing_out_file)

        key_cols = ["repo", "commit", "filepath"]

        # 🔥 Compare ONLY the newly processed rows
        before = len(df_merged)
        df_new = df_merged.merge(
            existing_df[key_cols], on=key_cols, how="left", indicator=True
        )

        df_new = df_new[df_new["_merge"] == "left_only"].drop(columns=["_merge"])
        after = len(df_new)

        logger.info(f"Skipping existing rows: {before - after} duplicates removed.")
        logger.info(f"Appending {after} new rows.")

        # Append new rows only
        final_df = pd.concat([existing_df, df_new], ignore_index=True)
        final_df.drop_duplicates(subset=key_cols, keep="last", inplace=True)

        # atomic_feather_save(final_df, out_file)
        final_df.to_feather(out_file)

        return

    print(f"[SAVE] Saving {len(df_merged)} rows to {out_file}")
    df_merged.to_feather(out_file)
    print("[DONE]")


# ---------------------------------------------------------------------
# Main transformation logic
# ---------------------------------------------------------------------
def transform(
    repos_filter: List[str] = None,
    workers: int = 8,
    skip_existing: bool = False,
    save_after: int = None,
    in_file: Path = JIT_FILE,
    existing_out_file: Path = JIT_FILE.with_name(
        JIT_FILE.stem + "_labeled_features_partial.feather"
    ),
    copy_out_file: bool = False,
):
    logger.info(f"[LOAD] {in_file}")
    logger.info(f"Using {workers} worker threads.")
    logger.info(f"Repos filter: {repos_filter}")

    # Load and filter DataFrame
    input_df = pd.read_feather(in_file)
    out_file = (
        existing_out_file.with_name(existing_out_file.stem + "_copy" + existing_out_file.suffix)
        if copy_out_file
        else existing_out_file
    )

    # if copy_out_file:
    #     out_file = out_file.with_name(out_file.stem + "_copy" + out_file.suffix)

    if repos_filter and len(repos_filter) > 0:
        logger.info(f"[FILTER] Limiting to repos: {repos_filter}")
        input_df = input_df[input_df["repo"].isin(repos_filter)]
        logger.info(f"Dataset size after filtering: {len(input_df)}")
    else:
        logger.info("No repository filter applied. Processing ALL repositories.")

    # CRITICAL ADDITION: Pre-calculate the expensive metrics
    logger.info("[PRECALC] Calculating author experience and recent activity...")
    # NOTE: You MUST ensure your input DF has an 'author' column (email) and 'datetime' column.
    input_df = pre_calculate_author_metrics(input_df, get_repo_func=get_repo_instance)
    logger.info("[PRECALC] Finished calculating author metrics.")

    if len(input_df) == 0:
        logger.warning("Dataset is empty after filtering. Exiting.")
        return

    if skip_existing:
        existing_df = in_file.with_name(
            in_file.stem + "_labeled_features_partial.feather"
        )
        if existing_df.exists():
            logger.info(
                f"skip_existing=True and {existing_df} exists → loading existing data..."
            )
            df_existing = pd.read_feather(existing_df)
            key_cols = ["repo", "commit", "filepath"]

            before = len(input_df)
            input_df = input_df.merge(
                df_existing[key_cols], on=key_cols, how="left", indicator=True
            )

            input_df = input_df[input_df["_merge"] == "left_only"]
            input_df = input_df.drop(columns=["_merge"])

            after = len(input_df)

            logger.info(
                f"Skipping already existing rows: {before - after} duplicates removed."
            )
            logger.info(f"{after} rows remain to process.")

            if len(input_df) == 0:
                print("[INFO] No new rows to process after skipping existing. Exiting.")
                return

    logger.info(f"Dataset size: {len(input_df)}")

    # try:
    #     extractor = FeatureExtractor(REPO_URL_MAP)
    # except Exception as e:
    #     print(f"[ERROR] Failed to initialize FeatureExtractor: {e}")
    #     return

    logger.info(f"[PROCESS] Starting feature extraction with {workers} workers...")

    # We need the 'repo' and 'commit' columns present in the input for merging later
    rows_to_process = list(
        input_df[["repo", "commit", "filepath"]].itertuples(index=False)
    )
    results_list = []

    classify_func = partial(classify_and_extract)

    # --- Parallel Execution Block with Ctrl+C Handling ---
    logger.info("Press Ctrl+C to stop processing and save partial results.")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Map original row tuples to futures
        futures = {
            executor.submit(classify_func, row, BUG_INDUCING_COMMITS[row.repo]): row
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
                            input_df=input_df,
                            results_list=results_list,
                            existing_out_file=existing_out_file,
                            append=skip_existing,
                            # copy_out_file=copy_out_file,
                            out_file=out_file
                        )
                        results_list.clear()
                        existing_out_file = out_file

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

            # The 'with' block will handle the final shutdown (wait=True)
            # The loop above will process any remaining completed futures before exiting.

    # --- Post-Processing and Saving ---

    if not results_list:
        # print(
        #     "[WARN] No results generated (or interrupted immediately). Exiting without saving."
        # )
        logger.warning(
            "No results generated (or interrupted immediately). Exiting without saving."
        )
        return

    # # Convert results back to a DataFrame
    # results_df = pd.DataFrame(results_list)

    # # 💡 CRITICAL CHANGE: Use merge for a SAFE partial save.
    # # This aligns the features/labels using the common keys ('repo', 'commit'),
    # # ensuring data integrity even if rows were skipped or completed out of order.
    # df_merged = df.merge(
    #     results_df,
    #     on=["repo", "commit", "filepath"],
    #     how="inner",  # Only keep rows that were successfully processed
    # )

    # df_merged["recent_churn"] = calc_recent_churn_from_df(df_merged, window_days=30)

    # If the same file already exists, I want to ask user to confirm append/overwrite
    # if they decide to append, scan existing file for last processed commits and skip those in df_merged if user agrees

    _save_to_file(
        input_df=input_df,
        results_list=results_list,
        existing_out_file=existing_out_file,
        append=skip_existing,
        # copy_out_file=copy_out_file,
        out_file=out_file
    )
    # if skip_existing and out_file.exists():
    #     print(f"[INFO] Append=True and {out_file} exists → loading existing data...")

    #     existing_df = pd.read_feather(out_file)

    #     # Detect rows already present → use the same keys to identify duplicates
    #     key_cols = ["repo", "commit", "filepath"]

    #     # Find new rows that aren't yet in the output
    #     before = len(df_merged)
    #     df_merged = df_merged.merge(
    #         existing_df[key_cols],
    #         on=key_cols,
    #         how="left",
    #         indicator=True
    #     )

    #     # Keep only rows NOT already in existing_df
    #     df_merged = df_merged[df_merged["_merge"] == "left_only"]
    #     df_merged = df_merged.drop(columns=["_merge"])

    #     after = len(df_merged)

    #     print(f"[INFO] Skipping already existing rows: {before - after} duplicates removed.")
    #     print(f"[INFO] Appending {after} new rows to existing dataset.")

    #     # Append and remove duplicates just in case
    #     final_df = pd.concat([existing_df, df_merged], ignore_index=True)
    #     final_df = final_df.drop_duplicates(subset=key_cols, keep="last")

    #     final_df.to_feather(out_file)
    #     print(f"[SAVE] Appended rows saved to {out_file}")
    #     print("[DONE]")
    #     return

    # print(f"[SAVE] Saving {len(df_merged)} rows to {out_file}")
    # df_merged.to_feather(out_file)
    # print("[DONE]")


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
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel classification",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Append to existing output file if it exists, skipping already processed commits.",
    )
    parser.add_argument(
        "--save-after",
        type=int,
        default=None,
        help="Stores after N processed rows. Defaults to no limit.",
    )

    parser.add_argument(
        "--copy-output",
        action="store_true",
        help="Save the output into a copy file.",
        required=False,
        default=False,
    )

    args = parser.parse_args()

    for repo in args.repos:
        if not is_registered(repo):
            raise ValueError(
                f"Invalid --repo value: {repo}. This repository is not registered."
            )

    if args.save_after < 1:
        raise ValueError(
            f"Invalid --save-after value: {args.save_after}. Must be a positive number (>0)."
        )

    load_bug_inducing_comms()

    transform(
        repos_filter=args.repos,
        workers=args.workers,
        skip_existing=args.skip_existing,
        save_after=args.save_after,
        copy_out_file=args.copy_output,
    )
