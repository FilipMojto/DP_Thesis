
import git
from argparse import ArgumentParser

from src_code.preprocessing.features.code_structural import extract_code_structural_features
from src_code.preprocessing.features.historical_temporal import calc_time_since_last_change
from src_code.preprocessing.features.linelevel import count_token_keywords
from src_code.preprocessing.features.semantic_embedding import calculate_semantic_embeddings
from src_code.preprocessing.features.textual_nlp import compute_msg_flags
from .features.change_churn import calculate_change_churn_metrics

### ---------- CONFIGURATION ----------
REPO_PATH = "./pandas/pandas"
MAPPING_FILE = "./bug_inducing_commits/pandas.yaml"
OUTPUT_FILE = "commit_features.csv"

parser = ArgumentParser(description="Extract features from commits in a Git repository.")
parser.add_argument("--repo", type=str, required=False, default=REPO_PATH)
parser.add_argument("--limit", type=int, required=False, default=None)


### ---------- HELPER FUNCTIONS ----------
def get_commit(repo: git.Repo, commit_hash: str) -> git.Commit | None:
    try:
        return repo.commit(commit_hash)
    except Exception:
        return None


### ---------- MAIN FEATURE EXTRACTION ----------
def extract_commit_features(repo, commit_hash):
    c = get_commit(repo, commit_hash)
    if not c:
        return None
    
    diff_text = c.diff(c.parents[0] if c.parents else None, create_patch=True)

    features = {"commit": commit_hash}
    features.update(calculate_change_churn_metrics(diff_text))

    # --- Commit message features ---
    features.update(compute_msg_flags(c.message))

    features.update(extract_code_structural_features(diff_text))

    features.update(calc_time_since_last_change(c))
    # features.update(calc_recent_churn(c))
    # features

    # --- Token keyword counts ---
    token_counts = count_token_keywords(diff_text)
    features.update(token_counts)

    features.update(calculate_semantic_embeddings(c, diff_text))


    return features
