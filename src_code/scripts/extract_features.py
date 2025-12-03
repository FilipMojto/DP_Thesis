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

# --- NEW HELPER FUNCTION (To be placed with other helper functions) ---

def get_functions_in_diff_range(code: str, changed_lines: set) -> list[str]:
    """
    Identifies and extracts the full source code of functions that intersect
    with the set of lines changed in the diff.
    """
    functions_to_analyze = []
    
    # 1. Parse the AST to find function/class definitions and their line numbers
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    # 2. Iterate over all nodes to find function/method definitions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # The function definition starts at node.lineno (1-based)
            # Find the end line number (requires using the source code or a helper)
            # For simplicity, we assume end is the last line of the function block.
            # A perfect solution would use astor or similar tools, but we approximate:
            
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', start_line + 1) # Fallback if no end_lineno
            
            # Check if the function body (including its signature) overlaps with changed lines
            if any(line in changed_lines for line in range(start_line, end_line + 1)):
                
                # Extract the source code of the function body
                lines = code.splitlines()
                # ast lines are 1-based, list indices are 0-based
                func_lines = lines[start_line - 1 : end_line] 
                functions_to_analyze.append('\n'.join(func_lines))
    
    return functions_to_analyze

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

        # 💡 Optimization Step 1: Skip non-Python files (optional but highly recommended)
        if not d.b_path.endswith('.py'):
            continue

        try:
            new_code = d.b_blob.data_stream.read().decode('utf-8', errors='ignore')
            old_code = d.a_blob.data_stream.read().decode('utf-8', errors='ignore') if d.a_blob else ""
        except Exception:
            continue

        changed_lines_new = set()

        patch_text = d.diff.decode(errors="ignore")
        
        # This is a basic way to get lines. A more robust way uses the diff library.
        # It relies on reading the hunk headers (@@ -old_start,old_count +new_start,new_count @@)
        
        for hunk_match in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", patch_text, re.MULTILINE):
            new_start = int(hunk_match.group(1))
            new_count = int(hunk_match.group(2) or 1)
            changed_lines_new.update(range(new_start, new_start + new_count))


        # 💡 Optimization Step 3: Analyze only the code of affected functions
        
        # Analyze NEW code
        new_functions = get_functions_in_diff_range(new_code, changed_lines_new)
        
        # Analyze OLD code (Need to find the corresponding old lines, which is complex. 
        # For simplicity in this fix, we analyze the *new* versions and the *old* versions
        # corresponding to the *newly identified* functions.)
        old_functions = []
        if old_code:
            # Re-run changed lines identification on the OLD code for robustness, 
            # though this is still tricky due to line shifts. A simpler, common 
            # approach is to assume the same function names were affected.
            # We skip detailed old line mapping here for brevity and focus on the main win:
            # analyzing smaller blocks of code.
            
            # For simplicity, we analyze the old version of the file and the functions identified by the new lines.
            old_functions = get_functions_in_diff_range(old_code, changed_lines_new)

        
        # Use the union of new and old functions to calculate delta
        
        new_total_ast = sum(count_ast_nodes(f) for f in new_functions)
        old_total_ast = sum(count_ast_nodes(f) for f in old_functions)
        
        new_total_complexity = sum(get_complexity(f) for f in new_functions)
        old_total_complexity = sum(get_complexity(f) for f in old_functions)

        ast_delta += abs(new_total_ast - old_total_ast)
        complexity_delta += abs(new_total_complexity - old_total_complexity)

        # Max function change calculation (can be simplified if only looking at snippets)
        if new_functions:
            max_func_change = max(max_func_change, max(len(f.splitlines()) for f in new_functions))
            
        # ast_delta += abs(count_ast_nodes(new_code) - count_ast_nodes(old_code))
        # complexity_delta += abs(get_complexity(new_code) - get_complexity(old_code))
        # max_func_change = max(max_func_change, len(new_code.splitlines()))

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