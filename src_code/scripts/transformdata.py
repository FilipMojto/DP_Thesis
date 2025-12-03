import argparse
import threading
from typing import List
import yaml
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from functools import lru_cache, partial
from tqdm import tqdm
import time
from git import Repo

from src_code.preprocessing.extraction import extract_commit_features
from src_code.preprocessing.features.developer_social import pre_calculate_author_metrics
from src_code.preprocessing.features.historical_temporal import calc_recent_churn_from_df
from src_code.utils.feature_extractor import FeatureExtractor 


from ..config import *


# ---------------------------------------------------------------------
# Load per-repo YAML files lazily and cache them
# ---------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_bug_inducing_for_repo(repo_name: str):
    """Load bug-inducing commits for a given repo based on filename."""

    yaml_files = list(BUG_INDUCING_DIR.glob(f"{repo_name}.*"))

    if not yaml_files:
        print(f"[WARN] No YAML found for repo: {repo_name}")
        return set()

    file_path = yaml_files[0] # only one expected

    with open(file_path, "r") as f:
        data = yaml.safe_load(f) or {}
        inducing_set = set()

        for _, inducing_list in data.items():
            inducing_set.update(inducing_list or [])

    return inducing_set

# Assuming you have a dictionary mapping repo names to URLs
REPO_URL_MAP = {
    "pandas": "https://github.com/pandas-dev/pandas.git"
}
# --- END CONFIG PLACEHOLDERS ---

BUG_INDUCING_COMMITS = {}

for repo in REPO_URL_MAP.keys():
    BUG_INDUCING_COMMITS[repo] = load_bug_inducing_for_repo(repo)

stop_event = threading.Event()

class StopProcessing(Exception):
    """Custom exception to stop processing in worker threads."""
    pass


# ---------------------------------------------------------------------
# Commit classifier (now also extracts features)
# ---------------------------------------------------------------------
def classify_and_extract(extractor: FeatureExtractor, row, bug_set: set):
    # 💡 CRITICAL: Check stop event before starting expensive work (e.g., git clone)
    if stop_event.is_set():
        # Raise our custom exception to cleanly exit the thread
        raise StopProcessing(f"Stop event set. Skipping {row.repo}/{row.commit}")
        
    # print(f"[PROCESS] Repo: {row.repo}, Commit: {row.commit}")
    repo = row.repo
    commit_hash = row.commit
    filepath = row.filepath
    bug_inducing_file = BUG_INDUCING_DIR / repo
    
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
    print(f"[TIMING] Feature extraction for {repo}/{commit_hash} took {extraction_end - extraction_start:.2f} seconds.")
    # Merge results, critically including 'repo' and 'commit' for safe merging later
    result = {"repo": repo, "commit": commit_hash, "filepath": filepath, "label": label, **features} 
    # print(f"Processed commit {commit_hash} in repo {repo}: label={label}")

    # with open(bug_inducing_file)

    return result

def get_repo_instance(repo_name: str) -> Repo:
    """Helper function to load the local Git Repo object."""
    # Assuming PYTHON_LIBS_DIR points to where repos are cloned
    return Repo(PYTHON_LIBS_DIR / repo_name)

# ---------------------------------------------------------------------
# Main transformation logic
# ---------------------------------------------------------------------
def transform(repos_filter: List[str] = None, workers: int = 8, skip_existing: bool = False):
    print(f"[LOAD] {JIT_FILE}")
    print(f"[INFO] Using {workers} worker threads.")
    print(f"[INFO] Repos filter: {repos_filter}")

    # Load and filter DataFrame
    df = pd.read_feather(JIT_FILE)
    
    if repos_filter and len(repos_filter) > 0:
        print(f"[FILTER] Limiting to repos: {repos_filter}")
        df = df[df["repo"].isin(repos_filter)]
        print(f"[INFO] Dataset size after filtering: {len(df)}")
    else:
        print("[INFO] No repository filter applied. Processing ALL repositories.")

    # CRITICAL ADDITION: Pre-calculate the expensive metrics
    print("[PRECALC] Calculating author experience and recent activity...")
    # NOTE: You MUST ensure your input DF has an 'author' column (email) and 'datetime' column.
    df = pre_calculate_author_metrics(df, get_repo_func=get_repo_instance)
    print("[PRECALC] Finished calculating author metrics.")

    if len(df) == 0:
        print("[WARN] Dataset is empty after filtering. Exiting.")
        return
    
    if skip_existing:
        existing_df = JIT_FILE.with_name(JIT_FILE.stem + "_labeled_features_partial.feather")
        if existing_df.exists():
            print(f"[INFO] skip_existing=True and {existing_df} exists → loading existing data...")
            df_existing = pd.read_feather(existing_df)
            key_cols = ["repo", "commit", "filepath"]

            before = len(df)
            df = df.merge(
                df_existing[key_cols],
                on=key_cols,
                how="left",
                indicator=True
            )

            df = df[df["_merge"] == "left_only"]
            df = df.drop(columns=["_merge"])

            after = len(df)

            print(f"[INFO] Skipping already existing rows: {before - after} duplicates removed.")
            print(f"[INFO] {after} rows remain to process.")

            if len(df) == 0:
                print("[INFO] No new rows to process after skipping existing. Exiting.")
                return 

    print("[INFO] Dataset size:", len(df))

    try:
        extractor = FeatureExtractor(REPO_URL_MAP)
    except Exception as e:
        print(f"[ERROR] Failed to initialize FeatureExtractor: {e}")
        return

    print(f"[PROCESS] Starting feature extraction with {workers} workers...")
    
    # We need the 'repo' and 'commit' columns present in the input for merging later
    rows_to_process = list(df[['repo', 'commit', 'filepath']].itertuples(index=False))
    results_list = []
    
    classify_func = partial(classify_and_extract, extractor)
    
    # --- Parallel Execution Block with Ctrl+C Handling ---
    print("[INFO] Press Ctrl+C to stop processing and save partial results.")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Map original row tuples to futures
        futures = {executor.submit(classify_func, row, BUG_INDUCING_COMMITS[row.repo]): row for row in rows_to_process}
 
        pbar = tqdm(total=len(rows_to_process), desc="Extracting Features", 
            unit="commit")

        try:
            # Use as_completed and tqdm for progress and interruptibility
            # for future in pbar:
            for future in as_completed(futures):
                pbar.update(1)

                if stop_event.is_set():
                    # If stop_event was set in a signal handler, break the loop
                    print("[INFO] Stop event set, breaking result collection loop.")
                    break 
                
                try:
                    result = future.result()
                    results_list.append(result)
                except StopProcessing as e:
                    print(f"[INFO] Worker stopped: {e}")

                except CancelledError:
                    pass
                except Exception as e:
                    row = futures.get(future, 'UNKNOWN')
                    print(f"\n[ERROR] A task failed for commit {row.repo}/{row.commit}: {e}")
                    raise e
            
        except KeyboardInterrupt:
            # Main thread receives Ctrl+C
            print("\nCtrl+C received! Setting stop_event and cancelling futures...")
            stop_event.set()
            pbar.close()
            
            # Cancel all remaining futures
            for future in futures:
                future.cancel()
                
            # The 'with' block will handle the final shutdown (wait=True)
            # The loop above will process any remaining completed futures before exiting.

    # --- Post-Processing and Saving ---


    if not results_list:
        print("[WARN] No results generated (or interrupted immediately). Exiting without saving.")
        return

    # Convert results back to a DataFrame
    results_df = pd.DataFrame(results_list)
    
    # 💡 CRITICAL CHANGE: Use merge for a SAFE partial save.
    # This aligns the features/labels using the common keys ('repo', 'commit'), 
    # ensuring data integrity even if rows were skipped or completed out of order.
    df_merged = df.merge(
        results_df, 
        on=["repo", "commit", "filepath"], 
        how='inner' # Only keep rows that were successfully processed
    )

    df_merged['recent_churn'] = calc_recent_churn_from_df(df_merged, window_days=30)



    # If the same file already exists, I want to ask user to confirm append/overwrite
    # if they decide to append, scan existing file for last processed commits and skip those in df_merged if user agrees


    out_file = JIT_FILE.with_name(JIT_FILE.stem + "_labeled_features_partial.feather")

    if skip_existing and out_file.exists():
        print(f"[INFO] Append=True and {out_file} exists → loading existing data...")

        existing_df = pd.read_feather(out_file)

        # Detect rows already present → use the same keys to identify duplicates
        key_cols = ["repo", "commit", "filepath"]

        # Find new rows that aren't yet in the output
        before = len(df_merged)
        df_merged = df_merged.merge(
            existing_df[key_cols],
            on=key_cols,
            how="left",
            indicator=True
        )

        # Keep only rows NOT already in existing_df
        df_merged = df_merged[df_merged["_merge"] == "left_only"]
        df_merged = df_merged.drop(columns=["_merge"])

        after = len(df_merged)

        print(f"[INFO] Skipping already existing rows: {before - after} duplicates removed.")
        print(f"[INFO] Appending {after} new rows to existing dataset.")

        # Append and remove duplicates just in case
        final_df = pd.concat([existing_df, df_merged], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=key_cols, keep="last")

        final_df.to_feather(out_file)
        print(f"[SAVE] Appended rows saved to {out_file}")
        print("[DONE]")
        return

    # ---------------------------------------------------------------------
    # OVERWRITE MODE
    # ---------------------------------------------------------------------
    print(f"[SAVE] Saving {len(df_merged)} rows to {out_file} (overwrite mode)")
    df_merged.to_feather(out_file)
    print("[DONE]")


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

    args = parser.parse_args()
    transform(repos_filter=args.repos, workers=args.workers, skip_existing=args.skip_existing)