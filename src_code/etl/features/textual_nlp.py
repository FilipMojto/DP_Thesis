from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm

from notebooks.logging_config import MyLogger

# from src_code.etl.ETL import get_repo_instance
from src_code.config import ETL_PATH_MAPPINGS, INTERIM_DATA_DIR, LOG_DIR
from src_code.etl.config import DEF_ETL_LOGGER

# from src_code.etl.integrate_cols import fetch_messages_from_git
from src_code.etl.utils import find_duplicates, get_repo_instance
from src_code.ml_pipeline.data_utils import load_df
from src_code.versioning import VersionedFileManager


def compute_msg_flags(msg):
    msg_lower = msg.lower()
    return {
        "msg_len": len(msg),
        "has_fix_kw": int("fix" in msg_lower),
        "has_bug_kw": int("bug" in msg_lower),
    }


# def fetch_single_message(repo_name, commit_hash):
#     """Helper to fetch one message from git."""
#     try:
#         repo_obj = get_repo_instance(repo_name)
#         return repo_obj.commit(commit_hash).message
#     except:
#         return ""


# def load_message_cache(path: Path) -> dict:
#     if not path.exists():
#         return {}

#     cache = {}
#     with path.open() as f:
#         for line in f:
#             obj = json.loads(line)
#             cache[(obj["repo"], obj["commit"])] = obj["message"]
#     return cache


# def parallel_fetch_messages(
#     df: pd.DataFrame, logger: MyLogger, cache_path: Path, batch_size: int = 50
# ) -> pd.DataFrame:
#     logger.log_check(" - Fetching messages in parallel...")
#     cache = load_message_cache(cache_path)
#     batch = []
#     logger.log_result(f"Found {len(cache)} items in cache. Finding duplicates...")
#     # unique_commits = df[["repo", "commit"]].drop_duplicates()
#     keys = ['repo', 'commit']
#     duplicates = find_duplicates(df=df, keys=keys)

#     if duplicates:
#         logger.log_result(f"Warning: Input DataFrame contains duplicates: {duplicates}")

#     unique_commits = (
#         df[keys]
#         .drop_duplicates()
#     )

    

#     to_fetch = [
#         tuple(x) for x in unique_commits.values
#         if tuple(x) not in cache
#     ]

#     logger.log_result(f"Found {len(unique_commits.values) - len(to_fetch)} duplicates!")

#     # with ThreadPoolExecutor(max_workers=8) as executor:
#     #     # Wrap the map in tqdm and provide the total explicitly
#     #     # converting the iterator to a list forces iteration and blocks until all tasks finish
#     #     # fetch_single_message runs while list() is being built
#     #     results = list(
#     #         tqdm.tqdm(
#     #             executor.map(lambda p: fetch_single_message(*p), unique_commits.values),
#     #             total=len(unique_commits),
#     #             desc="Git Fetch",
#     #         )
#     #     )

#     lock = threading.Lock()

#     with ThreadPoolExecutor(max_workers=8) as executor:
#         futures = {
#             executor.submit(fetch_single_message, repo, commit): (repo, commit)
#             for repo, commit in to_fetch
#         }

#         with cache_path.open("a") as f, tqdm.tqdm(total=len(futures), desc="Git Fetch") as pbar:
#             for future in as_completed(futures):
#                 repo, commit = futures[future]
#                 try:
#                     msg = future.result()
#                 except Exception as e:
#                     msg = ""

#                 # with lock:
#                 #     cache[(repo, commit)] = msg
#                 #     f.write(json.dumps({
#                 #         "repo": repo,
#                 #         "commit": commit,
#                 #         "message": msg
#                 #     }) + "\n")
#                 #     f.flush()
#                 with lock:
#                     cache[(repo, commit)] = msg
#                     batch.append({
#                         "repo": repo,
#                         "commit": commit,
#                         "message": msg
#                     })

#                 if len(batch) >= batch_size:
#                     for record in batch:
#                         f.write(json.dumps(record) + "\n")
#                     f.flush()
#                     batch.clear()
#                     logger.log_result("Flushing...", print_to_console=True)

#                 pbar.update(1)
            
#             if batch:
#                 for record in batch:
#                     f.write(json.dumps(record) + "\n")
#                 f.flush()

#     # mapping = dict(zip(map(tuple, unique_commits.values), results))
#     # df["message"] = df.apply(lambda x: mapping.get((x.repo, x.commit), ""), axis=1)
#     df['message'] = list(zip(df.repo, df.commit))
#     df['message'] = df['message'].map(cache).fillna("")

#     duplicates_after = find_duplicates(df=df, keys=keys)

#     # any new duplicates ?
#     new_duplicates = set(duplicates_after) - set(duplicates)

#     if new_duplicates:
#         logger.log_result(
#             f"Warning: Input DataFrame contains NEW duplicates: {sorted(new_duplicates)}"
#         )
#     return df


# import re


# def advanced_clean_msg(text):
#     # 1. Lowercase
#     text = text.lower()

#     # 2. Remove SVN metadata (extremely common in Pandas/NumPy history)
#     text = re.sub(r"git-svn-id:.*", "", text)

#     # 3. Remove URLs
#     text = re.sub(r"https?://\S+", "", text)

#     # 4. Remove Hex Hashes (4+ chars) and PR numbers (e.g., #1234)
#     text = re.sub(r"\b[0-9a-f]{4,}\b", "", text)
#     text = re.sub(r"#\d+", "", text)

#     # 5. Remove file extensions (keep the name, lose the .py/.cy)
#     text = re.sub(r"\.py|\.c|\.cpp|\.h", " ", text)

#     # 6. Remove non-alphabetic noise
#     text = re.sub(r"[^a-zA-Z\s]", " ", text)

#     return text.strip()


def calculate_tfidf_features(
    df: pd.DataFrame,
    text_col: str = "message",
    max_features: int = 100,
    logger: MyLogger = DEF_ETL_LOGGER,
) -> pd.DataFrame:
    """
    Fits a TF-IDF vectorizer on the text column and joins the
    resulting features back to the dataframe.
    """

    # 💡 FIX: Check if message column is missing and fetch it if needed
    if text_col not in df.columns:
        logger.logger.error(f"'{text_col}' column missing")
        raise ValueError(f"'{text_col}' column missing from DataFrame.")
        # df = fetch_messages_from_git(df, logger=logger, col_name=text_col)
        # df = parallel_fetch_messages(df, logger)

    logger.log_check(
        f"[TF-IDF] Extracting top {max_features} terms from '{text_col}'..."
    )
    df_original_shape = df.shape
    # Ensure no NaN values in the text column
    corpus = df[text_col].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),  # Capture phrases like "fix bug"
        preprocessor=advanced_clean_msg,
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Create feature names
    feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]

    # Convert to dense DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=feature_names, index=df.index
    )

    logger.log_result(
        f"[TF-IDF] Extracted {tfidf_df.shape[1]} features. "
        f"Original DF shape: {df_original_shape}, New DF shape: {df.shape[0], df.shape[1] + tfidf_df.shape[1]}"
    )

    return pd.concat([df, tfidf_df], axis=1)


# if __name__ == "__main__":
#     logger = MyLogger(label="EXTRACT MSG", section_name='Parallel Message Extractor', file_log_path=LOG_DIR / 'parallel_msg_extractor.log')
#     script_versioner = VersionedFileManager(src_dir=INTERIM_DATA_DIR, file_name='train_extended')
    
#     df_path = ETL_PATH_MAPPINGS['train']['current_newest']
#     df = load_df(df_file_path=df_path, logger=logger)

#     parallel_fetch_messages(df=df, logger=logger, cache_path=Path('test.jsonl'))
