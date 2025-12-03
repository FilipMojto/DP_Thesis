# import os
# import re
# import ast
# import time
# import datetime
# import git
# import pandas as pd
# import yaml
# from radon.complexity import cc_visit
# from collections import defaultdict
# from argparse import ArgumentParser



# # Optional: semantic embeddings
# try:
#     from sentence_transformers import SentenceTransformer
#     embed_model = SentenceTransformer('microsoft/codebert-base')
# except Exception:
#     embed_model = None
#     print("⚠️ Skipping embeddings: install `sentence-transformers` for semantic features.")


# ### ---------- CONFIGURATION ----------
# REPO_PATH = "./pandas/pandas"       # path to your local pandas repository
# MAPPING_FILE = "./bug_inducing_commits/pandas.yaml"   # <-- YAML file with your mapping
# OUTPUT_FILE = "commit_features.csv"

# parser = ArgumentParser(description="Extract features from commits in a Git repository.")
# parser.add_argument("--repo", type=str, required=False, default=REPO_PATH,
#                     help="Path to the local Git repository.")
# parser.add_argument("--limit", type=int, required=False, default=None,
#                     help="Limit the number of commits to process.")

# ### ---------- HELPER FUNCTIONS ----------
# def get_commit(repo, commit_hash):
#     try:
#         return repo.commit(commit_hash)
#     except Exception:
#         return None


# def count_ast_nodes(code):
#     """Counts AST node types in Python code."""
#     try:
#         tree = ast.parse(code)
#         return len(list(ast.walk(tree)))
#     except Exception:
#         return 0


# def get_complexity(code):
#     """Return cyclomatic complexity delta (radon)."""
#     try:
#         blocks = cc_visit(code)
#         return sum(b.complexity for b in blocks)
#     except Exception:
#         return 0


# def compute_msg_flags(msg):
#     msg_lower = msg.lower()
#     return {
#         "msg_len": len(msg),
#         "has_fix_kw": int("fix" in msg_lower),
#         "has_bug_kw": int("bug" in msg_lower)
#     }


# def compute_author_experience(repo, author_email, before_timestamp):
#     commits = list(repo.iter_commits('--all', author=author_email, until=before_timestamp))
#     return len(commits)


# def compute_recent_activity(repo, author_email, before_timestamp, days=30):
#     cutoff = datetime.datetime.fromtimestamp(before_timestamp, tz=datetime.timezone.utc) - datetime.timedelta(days=days)
#     commits = list(repo.iter_commits('--all', author=author_email, since=cutoff))
#     return len(commits)


# ### ---------- MAIN FEATURE EXTRACTION ----------
# def extract_commit_features(repo, commit_hash):
#     c = get_commit(repo, commit_hash)
#     if not c:
#         return None

#     features = {"commit": commit_hash}

#     # --- Change / churn metrics ---
#     diff_text = c.diff(c.parents[0] if c.parents else None, create_patch=True)
#     loc_added = loc_deleted = files_changed = hunks_count = 0
#     for d in diff_text:
#         if d.a_blob and d.b_blob:
#             files_changed += 1
#             patch = d.diff.decode(errors="ignore")
#             loc_added += patch.count('\n+') - patch.count('\n+++')
#             loc_deleted += patch.count('\n-') - patch.count('\n---')
#             hunks_count += patch.count('@@')

#     features.update({
#         "loc_added": loc_added,
#         "loc_deleted": loc_deleted,
#         "files_changed": files_changed,
#         "hunks_count": hunks_count
#     })

#     # --- Commit message features ---
#     features.update(compute_msg_flags(c.message))

#     # --- Code structural / complexity metrics ---
#     ast_delta = 0
#     complexity_delta = 0
#     max_func_change = 0

#     for d in diff_text:
#         if not d.b_path or not d.b_blob:
#             continue
#         try:
#             new_code = d.b_blob.data_stream.read().decode('utf-8', errors='ignore')
#             old_code = d.a_blob.data_stream.read().decode('utf-8', errors='ignore') if d.a_blob else ""
#         except Exception:
#             continue

#         ast_delta += abs(count_ast_nodes(new_code) - count_ast_nodes(old_code))
#         complexity_delta += abs(get_complexity(new_code) - get_complexity(old_code))
#         max_func_change = max(max_func_change, len(new_code.splitlines()))

#     features.update({
#         "ast_node_delta": ast_delta,
#         "complexity_delta": complexity_delta,
#         "max_func_change_size": max_func_change
#     })

#     # --- Developer / social metrics ---
#     author = c.author.email
#     features["author_exp"] = compute_author_experience(repo, author, c.committed_date)
#     features["author_recent_activity"] = compute_recent_activity(repo, author, c.committed_date)

#     # --- Temporal metrics ---
#     features["time_since_last_change"] = 0
#     if c.parents:
#         last_time = c.parents[0].committed_date
#         features["time_since_last_change"] = c.committed_date - last_time

#     # --- Embeddings (optional) ---
#     if embed_model:
#         code_text = "\n".join(
#             d.b_blob.data_stream.read().decode('utf-8', errors='ignore')[:2000]
#             for d in diff_text if d.b_blob
#         )
#         msg_text = c.message
#         features["code_embed"] = embed_model.encode([code_text])[0].tolist()
#         features["msg_embed"] = embed_model.encode([msg_text])[0].tolist()
#     else:
#         features["code_embed"] = None
#         features["msg_embed"] = None

#     return features


# ### ---------- MAIN PIPELINE ----------
# def main():
#     args = parser.parse_args()

#     repo = git.Repo(args.repo)

#     with open(MAPPING_FILE, "r", encoding="utf-8") as f:
#         mapping = yaml.safe_load(f)

#     # Limit number of commits if specified
#     if args.limit:
#         mapping = dict(list(mapping.items())[:args.limit])
#         print(f"⚙️ Limiting to {len(mapping)} commits from YAML.")

#     print(f"🔍 Extracting features for {sum(len(v)+1 for v in mapping.values())} commits...")

#     all_features = []
#     curr_index = 0

#     for fix_commit, inducing_list in mapping.items():
#         for commit_hash in [fix_commit] + inducing_list:
#             curr_index += 1
            
#             # print(f"Extracting {commit_hash}...")
#             print(f"Extracting commit {commit_hash} at index {curr_index}")
#             feats = extract_commit_features(repo, commit_hash)
#             if feats:
#                 feats["label"] = 1 if commit_hash in inducing_list else 0  # 1 = bug-inducing, 0 = fix
#                 all_features.append(feats)

#     df = pd.DataFrame(all_features)
#     df.to_csv(OUTPUT_FILE, index=False)
#     print(f"✅ Saved features to {OUTPUT_FILE}")


# if __name__ == "__main__":
#     main()

import os
import re
import ast
import time
import datetime
import warnings
import git
import pandas as pd
import yaml
import numpy as np
from radon.complexity import cc_visit
from collections import defaultdict
from argparse import ArgumentParser

# Optional: semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('microsoft/codebert-base')
except Exception:
    embed_model = None
    print("⚠️ Skipping embeddings: install `sentence-transformers` for semantic features.")


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


def count_ast_nodes(code):
    try:
        # tree = ast.parse(code)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
        return len(list(ast.walk(tree)))
    except Exception:
        return 0


def get_complexity(code):
    try:
        blocks = cc_visit(code)
        return sum(b.complexity for b in blocks)
    except Exception:
        return 0


def compute_msg_flags(msg):
    msg_lower = msg.lower()
    return {
        "msg_len": len(msg),
        "has_fix_kw": int("fix" in msg_lower),
        "has_bug_kw": int("bug" in msg_lower)
    }


def compute_author_experience(repo, author_email, before_timestamp):
    commits = list(repo.iter_commits('--all', author=author_email, until=before_timestamp))
    return len(commits)


def compute_recent_activity(repo, author_email, before_timestamp, days=30):
    cutoff = datetime.datetime.fromtimestamp(before_timestamp, tz=datetime.timezone.utc) - datetime.timedelta(days=days)
    commits = list(repo.iter_commits('--all', author=author_email, since=cutoff))
    return len(commits)


### ---------- TOKEN FEATURES ----------
def count_token_keywords(diff_text):
    """Count keywords like TODO, FIXME, try, except, raise in added lines."""
    tokens = {
        "todo": 0,
        "fixme": 0,
        "try": 0,
        "except": 0,
        "raise": 0,
    }
    for d in diff_text:
        patch = d.diff.decode(errors="ignore")
        for line in patch.splitlines():
            if line.startswith('+') and not line.startswith('+++'):  # only added lines
                l = line.lower()
                tokens["todo"] += l.count("todo")
                tokens["fixme"] += l.count("fixme")
                tokens["try"] += l.count("try")
                tokens["except"] += l.count("except")
                tokens["raise"] += l.count("raise")
    return tokens


### ---------- CONTEXT EMBEDDING ----------
def compute_context_embedding(diff_text):
    """Compute mean-pooled embedding of surrounding code context for added lines."""
    if not embed_model:
        return None

    contexts = []
    for d in diff_text:
        try:
            patch = d.diff.decode(errors="ignore")
        except Exception:
            continue

        lines = patch.splitlines()
        for i, line in enumerate(lines):
            if line.startswith('+') and not line.startswith('+++'):
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                snippet = '\n'.join(lines[start:end])
                contexts.append(snippet[:512])  # truncate context window

    if not contexts:
        return None

    # Embed and mean-pool
    embeddings = embed_model.encode(contexts, show_progress_bar=False)
    return np.mean(embeddings, axis=0).tolist()


### ---------- MAIN FEATURE EXTRACTION ----------
def extract_commit_features(repo, commit_hash):
    c = get_commit(repo, commit_hash)
    if not c:
        return None

    features = {"commit": commit_hash}

    # --- Change / churn metrics ---
    diff_text = c.diff(c.parents[0] if c.parents else None, create_patch=True)
    loc_added = loc_deleted = files_changed = hunks_count = 0
    for d in diff_text:
        if d.a_blob and d.b_blob:
            files_changed += 1
            patch = d.diff.decode(errors="ignore")
            loc_added += patch.count('\n+') - patch.count('\n+++')
            loc_deleted += patch.count('\n-') - patch.count('\n---')
            hunks_count += patch.count('@@')

    features.update({
        "loc_added": loc_added,
        "loc_deleted": loc_deleted,
        "files_changed": files_changed,
        "hunks_count": hunks_count
    })

    # --- Commit message features ---
    features.update(compute_msg_flags(c.message))

    # --- Code structural / complexity metrics ---
    ast_delta = 0
    complexity_delta = 0
    max_func_change = 0
    for d in diff_text:
        if not d.b_path or not d.b_blob:
            continue
        try:
            new_code = d.b_blob.data_stream.read().decode('utf-8', errors='ignore')
            old_code = d.a_blob.data_stream.read().decode('utf-8', errors='ignore') if d.a_blob else ""
        except Exception:
            continue

        ast_delta += abs(count_ast_nodes(new_code) - count_ast_nodes(old_code))
        complexity_delta += abs(get_complexity(new_code) - get_complexity(old_code))
        max_func_change = max(max_func_change, len(new_code.splitlines()))

    features.update({
        "ast_node_delta": ast_delta,
        "complexity_delta": complexity_delta,
        "max_func_change_size": max_func_change
    })

    # --- Developer / social metrics ---
    author = c.author.email
    features["author_exp"] = compute_author_experience(repo, author, c.committed_date)
    features["author_recent_activity"] = compute_recent_activity(repo, author, c.committed_date)

    # --- Temporal metrics ---
    features["time_since_last_change"] = 0
    if c.parents:
        last_time = c.parents[0].committed_date
        features["time_since_last_change"] = c.committed_date - last_time

    # --- Token keyword counts ---
    token_counts = count_token_keywords(diff_text)
    features.update(token_counts)

    # --- Context embeddings ---
    # features["context_embed"] = compute_context_embedding(diff_text)

    # --- Message and code embeddings ---
    if embed_model:
        code_text = "\n".join(
            d.b_blob.data_stream.read().decode('utf-8', errors='ignore')[:2000]
            for d in diff_text if d.b_blob
        )
        msg_text = c.message
        features["code_embed"] = embed_model.encode([code_text])[0].tolist()
        features["msg_embed"] = embed_model.encode([msg_text])[0].tolist()
    else:
        features["code_embed"] = None
        features["msg_embed"] = None

    return features


### ---------- MAIN PIPELINE ----------
def main():
    args = parser.parse_args()
    repo = git.Repo(args.repo)

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    if args.limit:
        mapping = dict(list(mapping.items())[:args.limit])
        print(f"⚙️ Limiting to {len(mapping)} commits from YAML.")

    print(f"🔍 Extracting features for {sum(len(v)+1 for v in mapping.values())} commits...")

    all_features = []
    curr_index = 0

    for fix_commit, inducing_list in mapping.items():
        for commit_hash in [fix_commit] + inducing_list:
            curr_index += 1
            print(f"Extracting commit {commit_hash} at index {curr_index}")
            feats = extract_commit_features(repo, commit_hash)
            if feats:
                feats["label"] = 1 if commit_hash in inducing_list else 0
                all_features.append(feats)

    df = pd.DataFrame(all_features)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved features to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()