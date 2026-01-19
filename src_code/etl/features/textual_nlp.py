from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm

from notebooks.logging_config import MyLogger
# from src_code.etl.ETL import get_repo_instance
from src_code.etl.config import DEF_ETL_LOGGER
# from src_code.etl.integrate_cols import fetch_messages_from_git
from src_code.etl.utils import get_repo_instance


def compute_msg_flags(msg):
    msg_lower = msg.lower()
    return {
        "msg_len": len(msg),
        "has_fix_kw": int("fix" in msg_lower),
        "has_bug_kw": int("bug" in msg_lower),
    }

# def fetch_messages_from_git(df: pd.DataFrame, logger: MyLogger, col_name: str = 'message') -> pd.DataFrame:
#     """
#     Iterates through the DataFrame and fetches the commit message 
#     from the local git repository for each commit hash.
#     """
#     logger.log_check("[GIT] Fetching commit messages from local clones...")
    
#     # We use a cache to avoid opening the same repo/commit multiple times
#     # if the same commit appears for multiple files.
#     message_map = {}
    
#     # Get unique repo/commit pairs to minimize git operations
#     unique_commits = df[['repo', 'commit']].drop_duplicates()
    
#     for _, row in tqdm.tqdm(unique_commits.iterrows(), total=len(unique_commits), desc="Fetching Messages"):
#         key = (row.repo, row.commit)
#         try:
#             repo_obj = get_repo_instance(row.repo)
#             msg = repo_obj.commit(row.commit).message
#             message_map[key] = msg
#         except Exception as e:
#             logger.error(f"Could not fetch message for {row.repo}/{row.commit}: {e}")
#             message_map[key] = ""

#     # Map the messages back to the main dataframe
#     df[col_name] = df.apply(lambda x: message_map.get((x.repo, x.commit), ""), axis=1)
#     return df

def fetch_single_message(repo_name, commit_hash):
    """Helper to fetch one message from git."""
    try:
        repo_obj = get_repo_instance(repo_name)
        return repo_obj.commit(commit_hash).message
    except:
        return ""

# def parallel_fetch_messages(df, logger: MyLogger) -> pd.DataFrame:
#     """Fetches git messages using multiple threads."""
#     logger.log_check(" - Fetching messages in parallel...")
    
#     # Get unique pairs to avoid redundant git calls
#     unique_commits = df[['repo', 'commit']].drop_duplicates()
    
#     # Use ThreadPoolExecutor for I/O bound git operations
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         results = list(tqdm(
#             executor.map(lambda p: fetch_single_message(*p), unique_commits.values),
#             total=len(unique_commits),
#             desc="Git Fetch"
#         ))
    
#     # Map back to the dataframe
#     mapping = dict(zip(map(tuple, unique_commits.values), results))
#     df['message'] = df.apply(lambda x: mapping.get((x.repo, x.commit), ""), axis=1)
#     return df
def parallel_fetch_messages(df, logger: MyLogger) -> pd.DataFrame:
    logger.log_check(" - Fetching messages in parallel...")
    unique_commits = df[['repo', 'commit']].drop_duplicates()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Wrap the map in tqdm and provide the total explicitly
        results = list(tqdm.tqdm(
            executor.map(lambda p: fetch_single_message(*p), unique_commits.values),
            total=len(unique_commits),
            desc="Git Fetch"
        ))
    
    mapping = dict(zip(map(tuple, unique_commits.values), results))
    df['message'] = df.apply(lambda x: mapping.get((x.repo, x.commit), ""), axis=1)
    return df

import re

def advanced_clean_msg(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove SVN metadata (extremely common in Pandas/NumPy history)
    text = re.sub(r'git-svn-id:.*', '', text)
    
    # 3. Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # 4. Remove Hex Hashes (4+ chars) and PR numbers (e.g., #1234)
    text = re.sub(r'\b[0-9a-f]{4,}\b', '', text)
    text = re.sub(r'#\d+', '', text)
    
    # 5. Remove file extensions (keep the name, lose the .py/.cy)
    text = re.sub(r'\.py|\.c|\.cpp|\.h', ' ', text)
    
    # 6. Remove non-alphabetic noise
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    return text.strip()

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
        # raise ValueError(f"'{text_col}' column missing from DataFrame.")
        # df = fetch_messages_from_git(df, logger=logger, col_name=text_col)
        df = parallel_fetch_messages(df, logger)

    logger.log_check(f"[TF-IDF] Extracting top {max_features} terms from '{text_col}'...")
    df_original_shape = df.shape
    # Ensure no NaN values in the text column
    corpus = df[text_col].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),  # Capture phrases like "fix bug"
        preprocessor=advanced_clean_msg
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
