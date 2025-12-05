# ---------------------------------------------------------------------
# Load per-repo YAML files lazily and cache them
# ---------------------------------------------------------------------
from functools import lru_cache

import yaml
from ..config import BUG_INDUCING_DIR

# Assuming you have a dictionary mapping repo names to URLs
REPO_URL_MAP = {
    "pandas": "https://github.com/pandas-dev/pandas.git",
    "airflow": "https://github.com/apache/airflow.git",
}
# --- END CONFIG PLACEHOLDERS ---

BUG_INDUCING_COMMITS = {}

# @lru_cache(maxsize=None)
def _load_bug_inducing_commits(repo_name: str):
    """Load bug-inducing commits for a given repo based on filename."""

    yaml_files = list(BUG_INDUCING_DIR.glob(f"{repo_name}.*"))

    if not yaml_files:
        print(f"[WARN] No YAML found for repo: {repo_name}")
        return set()

    file_path = yaml_files[0]  # only one expected

    with open(file_path, "r") as f:
        data = yaml.safe_load(f) or {}
        inducing_set = set()

        for _, inducing_list in data.items():
            inducing_set.update(inducing_list or [])

    return inducing_set


def load_bug_inducing_comms():
    for repo in REPO_URL_MAP.keys():
        BUG_INDUCING_COMMITS[repo] = _load_bug_inducing_commits(repo)


def is_registered(repo: str):
    return repo in REPO_URL_MAP