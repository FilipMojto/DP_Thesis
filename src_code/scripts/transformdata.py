# #!/usr/bin/env python3
# import argparse
# import threading
# import yaml
# from pathlib import Path
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from functools import lru_cache, partial
# import signal # 💡 NEW
# from tqdm import tqdm # 💡 NEW
# import time # 💡 NEW
# # DATA_DIR = Path("../../data")
# # DEFECTORS_DIR = DATA_DIR / "defectors"
# # JIT_FILE = DEFECTORS_DIR / "jit_bug_prediction_splits/time/train.feather"
# # BUG_INDUCING_DIR = DATA_DIR / "bug_inducing_commits"

# from ..config import *
# from ..utils.feature_extractor import FeatureExtractor
# # Assuming you have a dictionary mapping repo names to URLs
# REPO_URL_MAP = {
#     "openpilot": "https://github.com/commaai/openpilot.git",
#     "pandas": "https://github.com/pandas-dev/pandas.git",
#     # ... add all relevant repositories
# }

# stop_event = threading.Event()

# # Add this at the top of your script
# class StopProcessing(Exception):
#     """Custom exception to stop processing in worker threads."""
#     pass
# # ---------------------------------------------------------------------
# # Load per-repo YAML files lazily and cache them
# # ---------------------------------------------------------------------
# # @lru_cache(maxsize=None)
# # def load_repo_bug_map(repo_name: str):
# #     """Load bug-inducing-commits YAML for a given repo."""
# #     yaml_files = list((BUG_INDUCING_DIR / repo_name).glob("*.yml")) \
# #               + list((BUG_INDUCING_DIR / repo_name).glob("*.yaml"))

# #     if not yaml_files:
# #         print(f"[WARN] No YAML file found for repo: {repo_name}")
# #         return {}

# #     file_path = yaml_files[0]  # assume only one file per repo
# #     with open(file_path, "r") as f:
# #         data = yaml.safe_load(f)

# #     # Keys = fix commit hash, values = list of inducing commits
# #     # We want to gather *all* inducing commits
# #     inducing_set = set()
# #     for fix_hash, inducing_list in data.items():
# #         inducing_set.update(inducing_list or [])

# #     return inducing_set
# @lru_cache(maxsize=None)

# def load_bug_inducing_for_repo(repo_name: str):
#     """Load bug-inducing commits for a given repo based on filename."""

#     # Files like: airflow.yaml, airflow.yml, pandas.yaml...

#     yaml_files = list(BUG_INDUCING_DIR.glob(f"{repo_name}.*"))

#     if not yaml_files:

#         print(f"[WARN] No YAML found for repo: {repo_name}")
#         return set()



#     file_path = yaml_files[0] # only one expected

#     with open(file_path, "r") as f:
#         data = yaml.safe_load(f) or {}
#         inducing_set = set()

#         for fix_hash, inducing_list in data.items():

#             inducing_set.update(inducing_list or [])



#     return inducing_set


# # ---------------------------------------------------------------------
# # Commit classifier
# # ---------------------------------------------------------------------
# # def classify_commit(row):
# #     # repo = row["repo"]
# #     # commit_hash = row["hash"]
# #     repo = row.repo
# #     commit_hash = row.commit

# #     bug_set = load_repo_bug_map(repo)
# #     label = 1 if commit_hash in bug_set else 0
# #     return label

# # ---------------------------------------------------------------------
# # Commit classifier (now also extracts features)
# # ---------------------------------------------------------------------
# # Pass the extractor instance to the classifier
# def classify_and_extract(row, extractor: FeatureExtractor):
#     # 💡 REVISED: Check stop event before starting expensive work
#     if stop_event.is_set():
#         raise StopProcessing(f"Stop event set. Skipping {row.repo}/{row.commit}")
#     print(f"[PROCESS] Repo: {row.repo}, Commit: {row.commit}")
#     repo = row.repo
#     commit_hash = row.commit
    
#     # 1. Classification (Existing Logic)
#     bug_set = load_bug_inducing_for_repo(repo)
#     label = 1 if commit_hash in bug_set else 0

#     # 2. Feature Extraction (NEW Logic)
#     # This will handle cloning/pulling and feature calculation
#     features = extractor.extract_features(repo, commit_hash)
    
#     # Merge results
#     # result = {"label": label, **features}
#     # print(f"Processed commit {commit_hash} in repo {repo}: label={label}")
#     # Merge results
#     result = {"repo": repo, "commit": commit_hash, "label": label, **features} 
#     # ^ Added repo/commit back to result for safer partial merging
#     print(f"Processed commit {commit_hash} in repo {repo}: label={label}")
#     return result

# # ---------------------------------------------------------------------
# # Main transformation logic
# # ---------------------------------------------------------------------
# def transform(repos_filter=None, workers=8):
#     print(f"[LOAD] {JIT_FILE}")
#     print(f"[INFO] Using {workers} worker threads.")
#     print(f"[INFO] Repos filter: {repos_filter}")
#     df = pd.read_feather(JIT_FILE)

#     # if repos_filter:
#     #     print(f"[FILTER] Limiting to repos: {repos_filter}")
#     #     df = df[df["repo"].isin(repos_filter)]
#     # Check if a non-empty filter list was provided
#     if repos_filter and len(repos_filter) > 0:
#         print(f"[FILTER] Limiting to repos: {repos_filter}")
#         df = pd.read_feather(JIT_FILE)
#         df = df[df["repo"].isin(repos_filter)]
#     else:
#         # This handles repos_filter being None OR an empty list []
#         print("[INFO] No repository filter applied. Processing ALL repositories.")
#         df = pd.read_feather(JIT_FILE)

#     print("[INFO] Dataset size:", len(df))

#     # # Parallel commit classification
#     # print(f"[CLASSIFY] Starting with {workers} workers...")
#     # with ThreadPoolExecutor(max_workers=workers) as executor:
#     #     df["label"] = list(executor.map(classify_commit, df.itertuples(index=False)))

#     # out_file = JIT_FILE.with_name(JIT_FILE.stem + "_labeled.feather")
#     # print(f"[SAVE] {out_file}")
#     # df.reset_index(drop=True).to_feather(out_file)
#     # print("[DONE]")
#     # 💡 NEW: Initialize the feature extractor
#     # extractor = FeatureExtractor(REPO_URL_MAP)
#     # 💡 NEW: Initialize the feature extractor ONCE
#     try:
#         extractor = FeatureExtractor(REPO_URL_MAP)
#     except Exception as e:
#         print(f"[ERROR] Failed to initialize FeatureExtractor (cloning/pulling): {e}")
#         return # Exit if setup fails

#     print(f"[PROCESS] Starting feature extraction with {workers} workers...")
    
#     # Use itertuples for efficient iteration
#     rows_to_process = list(df.itertuples(index=False))
    
#     # with ThreadPoolExecutor(max_workers=workers) as executor:
#     #     # Use partial to pass the extractor instance to the function
#     #     from functools import partial
#     #     classify_func = partial(classify_and_extract, extractor=extractor)
        
#     #     # This returns a list of dictionaries (features + label)
#     #     results_list = list(executor.map(classify_func, rows_to_process))
#     results_list = []
#     futures = []

    
#     # Use partial to pass the extractor instance to the function
#     classify_func = partial(classify_and_extract, extractor=extractor)
    
#     # --- Parallel Execution Block with Ctrl+C Handling ---
#     # try:
#     print("[INFO] Press Ctrl+C to stop processing and save partial results.")
#     with ThreadPoolExecutor(max_workers=workers) as executor:
 
#         # Use executor.map which returns an iterator
#         # future_results = executor.map(classify_func, rows_to_process)
        
#         # # Iterate through the results to collect them. This loop is interruptible.
#         # # for result in future_results:
#         # #     results_list.append(result)
#         # for result in tqdm(
#         #     future_results, 
#         #     desc="Extracting Features", 
#         #     total=len(rows_to_process),
#         #     unit="commit"
#         # ):
#         #     results_list.append(result)
#         # submit all tasks
#         # for row in rows_to_process:
#         #     if stop_event.is_set():
#         #         print("[INFO] Stop event set, breaking task submission.")
#         #         break

#         #     futures.append(executor.submit(classify_func, row))
        
#         # futures = [executor.submit(classify_func, row) for pos, row in enumerate(rows_to_process)]
#         futures = {executor.submit(classify_func, row): row for row in rows_to_process}

#         # try:
#         #     while futures:
#         #         # iterate as futures complete (interruptible!)
#         #         for future in futures[:]:
#         #             if future.done():
#         #                 try:
#         #                     result = future.result()
#         #                     results_list.append(result)
#         #                 except Exception as e:
#         #                     print(f"[ERROR] A task failed: {e}")
#         #                 futures.remove(future)
#         #         time.sleep(0.5)  # small sleep to prevent busy waiting
#         # except KeyboardInterrupt:
#         #     print("Ctrl+C received! Setting stop_event...")
#         #     stop_event.set()
#         #     # Wait for threads to finish
#         #     for future in futures:
#         #         future.cancel()  # may not stop streaming_bulk immediately
#         #     executor.shutdown(wait=True)

#         try:
#             # Use tqdm and as_completed for progress bar and interruptibility
#             for future in tqdm(
#                 as_completed(futures), 
#                 total=len(rows_to_process), 
#                 desc="Extracting Features", 
#                 unit="commit"
#             ):
#                 if stop_event.is_set():
#                     # If stop_event is set (e.g., in a background signal handler)
#                     break 
                
#                 try:
#                     result = future.result()
#                     results_list.append(result)
#                 except StopProcessing as e:
#                     # Ignore our intentional stop signal from the worker
#                     print(f"\n[INFO] Worker stopped: {e}")
#                 except Exception as e:
#                     print(f"\n[ERROR] A task failed for commit {futures.get(future, 'UNKNOWN')}: {e}")
#                     # You might want to log this or handle it more gracefully
            
#         except KeyboardInterrupt:
#             print("\nCtrl+C received! Setting stop_event and cancelling futures...")
#             stop_event.set()
            
#             # Cancel all remaining futures
#             for future in futures:
#                 future.cancel()
                
#             # Need to collect results from futures that completed *before* the cancel/shutdown
#             # Loop over remaining futures (already processed by as_completed up to this point)
#             # This is automatically handled by the main loop structure above if we let the 
#             # 'with' statement exit naturally after setting the stop_event.

#             # # iterate as futures complete (interruptible!)
#             # for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting Features"):
#             #     try:
#             #         results_list.append(fut.result())
#             #     except Exception as e:
#             #         print(f"[ERROR] A task failed: {e}")

#     # except KeyboardInterrupt:
#     #     print("\n[STOP] KeyboardInterrupt detected.")
#     #     print("[STOP] Cancelling pending tasks...")

#     #     # Cancel all futures that have not started
#     #     for fut in futures:
#     #         fut.cancel()

#     #     print("[STOP] Saving partial results...")

#         # The 'with' statement handles the shutdown of the executor automatically (wait=True)
#         # We need to manually stop the ongoing map operation if necessary, 
#         # but collecting the list of results processed so far is the key.
    
#     # --- Post-Processing and Saving ---

#     if not results_list:
#         if len(df) > 0:
#              print("[WARN] No results generated. Exiting without saving.")
#         return

#     # Convert results back to a DataFrame for merging
#     # results_df = pd.DataFrame(results_list)

#     # 💡 NEW: Merge results with the original DataFrame
#     # df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

#     # out_file = JIT_FILE.with_name(JIT_FILE.stem + "_labeled_features.feather")
#     # print(f"[SAVE] {out_file}")
#     # df.reset_index(drop=True).to_feather(out_file)
#     # print("[DONE]")
#     df_partial = df.iloc[:len(results_list)].reset_index(drop=True)
    
#     # Merge results with the truncated original DataFrame
#     df_labeled = pd.concat([df_partial, results_list], axis=1) # Note: results_list is already a list of dicts

#     out_file = JIT_FILE.with_name(JIT_FILE.stem + "_labeled_features_partial.feather") # Changed name to reflect partial save
#     print(f"[SAVE] Saving {len(df_labeled)} rows to {out_file}")
#     df_labeled.to_feather(out_file)
#     print("[DONE]")


# # ---------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Transform JIT dataset with labels.")
#     parser.add_argument(
#         "--repos",
#         nargs="*",
#         help="List of repositories to include (default: all). Example: --repos airflow pandas",
#     )
#     parser.add_argument(
#         "--workers",
#         type=int,
#         default=8,
#         help="Number of worker threads for parallel classification",
#     )

#     args = parser.parse_args()
#     transform(repos_filter=args.repos, workers=args.workers)

#!/usr/bin/env python3
import argparse
import threading
import yaml
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from functools import lru_cache, partial
import signal 
from tqdm import tqdm
import time
from git import Repo

from src_code.utils.feature_extractor import FeatureExtractor 

# --- CONFIG PLACEHOLDERS (Replace with your actual imports) ---
# Assuming these are defined in your environment/config.py
from ..config import *

# Optional: semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('microsoft/codebert-base')
except Exception:
    embed_model = None
    print("⚠️ Skipping embeddings: install `sentence-transformers` for semantic features.")


# class FeatureExtractor:
#     """Mock class for demonstration purposes. Replace with your actual FeatureExtractor."""
#     def __init__(self, repo_map):
#         self.repo_map = repo_map
#         # Simulate setup time
#         print("[INIT] FeatureExtractor initialized.")
#         pass
    
#     def extract_features(self, repo, commit):
#         # Simulate work, which can be long and needs to be interruptible
#         # The main git operations here are the expensive parts.
#         time.sleep(0.01) # Small delay to see progress bar move
#         return {"feature_1": len(commit), "feature_2": repo.count('p')}

# Assuming you have a dictionary mapping repo names to URLs
REPO_URL_MAP = {
    # "openpilot": "https://github.com/commaai/openpilot.git",
    "pandas": "https://github.com/pandas-dev/pandas.git"
    # ... add all relevant repositories
}
# --- END CONFIG PLACEHOLDERS ---

stop_event = threading.Event()

class StopProcessing(Exception):
    """Custom exception to stop processing in worker threads."""
    pass

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

        for fix_hash, inducing_list in data.items():
            inducing_set.update(inducing_list or [])

    return inducing_set


# ---------------------------------------------------------------------
# Commit classifier (now also extracts features)
# ---------------------------------------------------------------------
def classify_and_extract(row, extractor: FeatureExtractor):
    from .extract_features import extract_commit_features
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
    bug_set = load_bug_inducing_for_repo(repo)
    print(repo, "bug set size:", len(bug_set))

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

# ---------------------------------------------------------------------
# Main transformation logic
# ---------------------------------------------------------------------
def transform(repos_filter=None, workers=8):
    print(f"[LOAD] {JIT_FILE}")
    print(f"[INFO] Using {workers} worker threads.")
    print(f"[INFO] Repos filter: {repos_filter}")

    # Load and filter DataFrame
    df = pd.read_feather(JIT_FILE)
    if repos_filter and len(repos_filter) > 0:
        print(f"[FILTER] Limiting to repos: {repos_filter}")
        df = df[df["repo"].isin(repos_filter)]
        df = df[8000:]  # TEMPORARY: PROCESS A SUBSET FOR TESTING

        print(f"[INFO] Dataset size after filtering: {len(df)}")
    else:
        print("[INFO] No repository filter applied. Processing ALL repositories.")

    if len(df) == 0:
        print("[WARN] Dataset is empty after filtering. Exiting.")
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
    
    classify_func = partial(classify_and_extract, extractor=extractor)
    
    # --- Parallel Execution Block with Ctrl+C Handling ---
    print("[INFO] Press Ctrl+C to stop processing and save partial results.")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Map original row tuples to futures
        futures = {executor.submit(classify_func, row): row for row in rows_to_process}
        # Initialize tqdm outside the for loop
        # pbar = tqdm(
        #     as_completed(futures), 
        #     total=len(rows_to_process), 
        #     desc="Extracting Features", 
        #     unit="commit"

        # )
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
                    # Catch the intentional stop signal from the worker thread
                    # print(f"\n[INFO] Worker stopped: {e}")
                    pass
                except CancelledError:
                    # Catch tasks cancelled by the main thread after Ctrl+C
                    pass
                except Exception as e:
                    # Catch any unexpected errors in the thread
                    row = futures.get(future, 'UNKNOWN')
                    print(f"\n[ERROR] A task failed for commit {row.repo}/{row.commit}: {e}")
            
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

    out_file = JIT_FILE.with_name(JIT_FILE.stem + "_labeled_features_partial.feather")
    print(f"[SAVE] Saving {len(df_merged)} rows to {out_file}")
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

    args = parser.parse_args()
    transform(repos_filter=args.repos, workers=args.workers)