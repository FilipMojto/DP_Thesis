# import argparse
# import time
# from typing import Dict
# import pandas as pd
# from tqdm import tqdm
# from notebooks.logging_config import MyLogger
# from src_code.config import ETL_PATH_MAPPINGS, INTERIM_DATA_DIR, LOG_DIR
# from src_code.etl.features.textual_nlp import calculate_tfidf_features
# from src_code.etl.utils import get_repo_instance
# from src_code.versioning import VersionedFileManager
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
from typing import get_args

import pandas as pd
import tqdm
from notebooks.logging_config import MyLogger
from src_code.config import (
    CACHE_DIR,
    ETL_PATH_MAPPINGS,
    EXTENDED_DATA_DIR,
    INTERIM_DATA_DIR,
    LOG_DIR,
    SubsetType,
)
from src_code.etl.utils import find_duplicates, get_repo_instance
from src_code.ml_pipeline.data_utils import load_df, save_df
from src_code.versioning import VersionedFileManager

stop_event = threading.Event()


def load_message_cache(path: Path) -> dict:
    if not path.exists():
        return {}

    cache = {}
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            cache[(obj["repo"], obj["commit"])] = obj["message"]
    return cache


def fetch_single_message(repo_name, commit_hash):
    """Helper to fetch one message from git."""
    if stop_event.is_set():
        return None

    try:
        repo_obj = get_repo_instance(repo_name)
        return repo_obj.commit(commit_hash).message
    except:
        return ""


def parallel_fetch_messages(
    df: pd.DataFrame, logger: MyLogger, cache_path: Path, batch_size: int = 50
) -> pd.DataFrame:
    """Extends a DataFrame with commit messages by fetching them from locally cloned repositories.

    Commit messages are stored in a persistent cache (JSONL file) and are updated in batches.
    If the program is interrupted, cached messages are preserved on disk, allowing processing
    to resume without repeating work. The DataFrame itself is updated only after all messages
    have been fetched.

    The primary purpose of this function is to prefetch commit messages in advance before
    computing their TF-IDF scores or other downstream text-based features.

    Args:
        df (pd.DataFrame): DataFrame to extend; a new column 'message' will be added.
        logger (MyLogger): Logger instance to record progress and events.
        cache_path (Path): Path to a JSONL file used as a persistent cache of already fetched messages.
        batch_size (int, optional): Number of messages to flush to disk at a time. Defaults to 50.

    Returns:
        pd.DataFrame: A copy of the original DataFrame extended with the 'message' column.

    """

    logger.log_check(" - Fetching messages in parallel...")
    cache = load_message_cache(cache_path)
    batch = []
    logger.log_result(f"Found {len(cache)} items in cache. Finding duplicates...")
    keys = ["repo", "commit"]
    duplicates = find_duplicates(df=df, keys=keys)

    if duplicates:
        logger.log_result(f"Warning: Input DataFrame contains duplicates: {duplicates}")

    unique_commits = df[keys].drop_duplicates()

    # NOTE: if cache contains some rows not included in the dataframe, they are ignored
    to_fetch = [tuple(x) for x in unique_commits.values if tuple(x) not in cache]

    logger.log_result(
        f"Found {len(unique_commits.values) - len(to_fetch)} duplicates in cache!"
    )
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_single_message, repo, commit): (repo, commit)
            for repo, commit in to_fetch
        }

        pbar = tqdm.tqdm(total=len(futures), desc="Git Fetch")

        with cache_path.open("a") as f:
            try:
                for future in as_completed(futures):
                    pbar.update(1)

                    if stop_event.is_set():
                        logger.info("Stop event set, breaking result collection loop.")
                        break

                    repo, commit = futures[future]
                    try:
                        msg = future.result()
                    except Exception as e:
                        msg = ""

                    with lock:
                        cache[(repo, commit)] = msg
                        batch.append({"repo": repo, "commit": commit, "message": msg})

                    if len(batch) >= batch_size:
                        for record in batch:
                            f.write(json.dumps(record) + "\n")
                        f.flush()
                        batch.clear()
                        logger.log_result("Flushing...", print_to_console=True)
            except KeyboardInterrupt:
                logger.log_result("⚠️ Interrupted by user. Stopping workers...", True)
                stop_event.set()
                pbar.close()

                # cancel futures that haven't started
                for future in futures:
                    future.cancel()

            finally:
                # flush remaining batch
                if batch:
                    logger.log_result(
                        f"Flushing remaining {len(batch)} records...", True
                    )
                    with cache_path.open("a") as f:
                        for record in batch:
                            f.write(json.dumps(record) + "\n")
                        f.flush()

    df["message"] = list(zip(df.repo, df.commit))
    df["message"] = df["message"].map(cache).fillna("")

    duplicates_after = find_duplicates(df=df, keys=keys)

    # any new duplicates ?
    new_duplicates = set(duplicates_after) - set(duplicates)

    if new_duplicates:
        logger.log_result(
            f"Warning: Input DataFrame contains NEW duplicates: {sorted(new_duplicates)}"
        )
    return df

# # ALL functions should take in a DataFrame and return an extended DataFrame
import time
from src_code.etl.features.textual_nlp import calculate_tfidf_features


# COLS_TO_INTEGRATE = {"tfifd_message": calculate_tfidf_features}
# COLS_TO_INTEGRATE = {"tfifd_message": {
#     "prefetch": {
#         "col": "message",
#         "call": lambda : parallel_fetch_messages()
#     },
#     "integrate": calculate_tfidf_features
# }}

# DF_TO_EXTEND = [
#     ETL_PATH_MAPPINGS["train"]["current_newest"],
#     ETL_PATH_MAPPINGS["test"]["current_newest"],
#     ETL_PATH_MAPPINGS["validate"]["current_newest"],
# ]


def integrate_additional_columns(input_df: pd.DataFrame, logger: MyLogger, subset: str = 'train', max_rows: int | None = None):
    # db_file_versioner = VersionedFileManager(src_dir=INTERIM_DATA_DIR, file_name=subset + "_extended.feather")

    # # for df_info in DF_TO_EXTEND:
    # input_path = ETL_PATH_MAPPINGS[subset]['current_newest']
    # print(f"Integrating additional columns into {input_path}...")

    # logger.log_check(f"Integrating additional columns into {input_path}...")
    # df = pd.read_feather(input_path)

    if max_rows is not None:
        input_df = input_df.head(max_rows)

    # for col_prefix, body in COLS_TO_INTEGRATE.items():

    #     if body['prefetch']['col'] not in df.columns:
    #         body["prefetch"]["call"]()

    start = time.time()
    # logger.log_check(f" - Integrating columns with prefix '{col_prefix}'...")
    # df = body(df, text_col="message", logger=logger)

    # if 'message' not in 
    input_df = calculate_tfidf_features(df=input_df, text_col='message', max_features=100, logger=logger)

    end = time.time()
    logger.log_result(f"   -> Completed in {end - start:.2f} seconds.")

    # output_path = input_path.parent / (input_path.stem + "_extended.feather")
    # output_path = db_file_versioner.next_base_output

    # logger.log_result(f" - Saving extended dataframe to {output_path}...")
    # input_df.reset_index(drop=True).to_feather(output_path)
    input_df = input_df.reset_index(drop=True)
    return input_df


# # class BatchUpdater():

# #     def __init__(self, original_df: pd.DataFrame, extended_df: pd.DataFrame = None, save_after: int = 500):
# #         self.original_df = original_df
# #         self.extended_df = extended_df if extended_df else pd.DataFrame()
# #         self.buffer = []
# #         self.save_after = save_after


# #     def add_row(self, row: Dict):
# #         self.buffer.append(row)

# #         if self.buffer >= self.save_after:
# #             results_df = pd.DataFrame(self.buffer)

# #             # 💡 CRITICAL CHANGE: Use merge for a SAFE partial save.
# #             # This aligns the features/labels using the common keys ('repo', 'commit'),
# #             # ensuring data integrity even if rows were skipped or completed out of order.
# #             df_merged = self.original_df.merge(
# #                 results_df,
# #                 on=["repo", "commit", "filepath"],
# #                 how="inner",  # Only keep rows that were successfully processed
# #             )


# if __name__ == "__main__":
#     argsparser = argparse.ArgumentParser(description="Integrate additional columns into ETL DataFrames.")
#     argsparser.add_argument("--max-rows", type=int, required=False, default=None, help="Maximum number of rows to process from each DataFrame.")
#     argsparser.add_argument("--subset", type=str, choices=["train", "test", "validate"], required=False, default='train', help="Subset to process (train, test, validate). If not specified, processes all.")

#     args = argsparser.parse_args()


#     SCRIPT_LOGGER = MyLogger(
#         label="ETL",
#         section_name="Column Integration",
#         file_log_path=LOG_DIR / "etl_column_integration.log",
#     )

#     SCRIPT_LOGGER.log_check("Starting column integration process...")
#     integrate_additional_columns(logger=SCRIPT_LOGGER, max_rows=args.max_rows, subset=args.subset)
#     SCRIPT_LOGGER.log_result("Column integration process completed.")


DEF_MAX_ROWS = 100


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--subset",
        choices=get_args(SubsetType),
        required=True,
        help=f"Specify the subset ({get_args(SubsetType)}) to process.",
    )
    argparser.add_argument(
        "--prefetch-messages",
        action='store_true',
        help="Should messages be prefetched (required for tfidf)?",
        required=False,
        default=False
    )
    argparser.add_argument(
        "--max-rows",
        type=int,
        required=False,
        default=None,
        help="Limit to the first n rows to process."
    )


    args = argparser.parse_args()

    subset: SubsetType = args.subset
    prefetch_messages: bool = args.prefetch_messages
    max_rows: int = args.max_rows

    cached_messages = CACHE_DIR / f"{subset}_commit_messages.jsonl"

    logger = MyLogger(
        label="EXTRACT MSG",
        section_name="Parallel Message Extractor",
        file_log_path=LOG_DIR / "parallel_msg_extractor.log",
    )
    input_df_versioner = VersionedFileManager(
        file_path=INTERIM_DATA_DIR / f"{subset}_labeled_features_partial.feather"
    )

    output_df_versioner = VersionedFileManager(
        file_path=EXTENDED_DATA_DIR / f"{subset}_extended.feather"
    )

    # df_path = ETL_PATH_MAPPINGS["test"]["current_newest"]
    # input_df = load_df(df_file_path=input_df_versioner.current_newest, logger=logger)
    # keys = ["repo", "commit"]

    # df_size_before = len(input_df)
    # input_df = input_df.drop_duplicates(subset=keys)
    # df_size_after = len(input_df)

    # logger.log_result(
    #     f"Dropped {df_size_before - df_size_after} ({(df_size_before - df_size_after) / df_size_before}%) duplicates!"
    # )

    # if prefetch_messages:
    input_df = load_df(df_file_path=input_df_versioner.current_newest, logger=logger)
    keys = ["repo", "commit"]

    df_size_before = len(input_df)
    input_df = input_df.drop_duplicates(subset=keys)
    df_size_after = len(input_df)

    logger.log_result(
        f"Dropped {df_size_before - df_size_after} ({(df_size_before - df_size_after) / df_size_before}%) duplicates!"
    )

    input_df = parallel_fetch_messages(df=input_df, logger=logger, cache_path=cached_messages)
    # else:
    #     input_df = load_df(df_file_path=output_df_versioner.current_newest, logger=logger)

    # input_df = integrate_additional_columns(input_df=input_df, logger=logger, subset=subset, max_rows=max_rows)

    save_df(df=input_df, df_file_path=output_df_versioner.next_base_output, logger=logger)
