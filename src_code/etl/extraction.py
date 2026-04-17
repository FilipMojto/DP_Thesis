
import git
from argparse import ArgumentParser

from .features.code_structural import extract_code_structural_features
from .features.historical_temporal import calc_time_since_last_change
from .features.linelevel import count_token_keywords
from .features.semantic_embedding import calculate_semantic_embeddings
from .features.textual_nlp import compute_msg_flags
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
# def extract_commit_features(repo, commit_hash):
#     c = get_commit(repo, commit_hash)
    
#     if not c:
#         return None
    
#     # diff_text = c.diff(c.parents[0] if c.parents else None, create_patch=True)
#     parent = c.parents[0] if c.parents else None
#     diff_text = parent.diff(c, create_patch=True) if parent else c.diff(git.NULL_TREE, create_patch=True)

#     features = {"commit": commit_hash}
#     features.update(calculate_change_churn_metrics(diff_text))

#     # --- Commit message features ---
#     features.update(compute_msg_flags(c.message))

#     features.update(extract_code_structural_features(diff_text))

#     features.update(calc_time_since_last_change(c=c))
#     # features.update(calc_recent_churn(c))
#     # features

#     # --- Token keyword counts ---
#     token_counts = count_token_keywords(diff_text)
#     features.update(token_counts)

#     features.update(calculate_semantic_embeddings(c, diff_text))


#     return features
def extract_commit_features(repo, commit_hash, logger):
    c = get_commit(repo, commit_hash)

    if not c:
        return None

    features = {"commit": commit_hash}

    try:
        diff_text = c.diff(c.parents[0] if c.parents else None, create_patch=True)
    except Exception as e:
        logger.error(f"[DIFF ERROR] {repo.working_dir}/{commit_hash}: {e}")
        return features

    try:
        churn = calculate_change_churn_metrics(diff_text)
        if churn:
            features.update(churn)
    except Exception as e:
        logger.error(f"[CHURN ERROR] {repo.working_dir}/{commit_hash}: {e}")

    try:
        msg_flags = compute_msg_flags(c.message)
        if msg_flags:
            features.update(msg_flags)
    except Exception as e:
        logger.error(f"[MSG FLAGS ERROR] {repo.working_dir}/{commit_hash}: {e}")

    try:
        structural = extract_code_structural_features(diff_text)
        if structural:
            features.update(structural)
    except Exception as e:
        logger.error(f"[STRUCTURAL ERROR] {repo.working_dir}/{commit_hash}: {e}")

    try:
        temporal = calc_time_since_last_change(repo=repo, c=c)
        if temporal:
            features.update(temporal)
    except Exception as e:
        logger.error(f"[TEMPORAL ERROR] {repo.working_dir}/{commit_hash}: {e}")

    try:
        token_counts = count_token_keywords(diff_text)
        if token_counts:
            features.update(token_counts)
    except Exception as e:
        logger.error(f"[TOKEN ERROR] {repo.working_dir}/{commit_hash}: {e}")

    try:
        semantic = calculate_semantic_embeddings(c, diff_text)
        if semantic:
            features.update(semantic)
    except Exception as e:
        logger.error(f"[EMBED ERROR] {repo.working_dir}/{commit_hash}: {e}")

    return features
