# ---------------------------------------------------------------------
# Load per-repo YAML files lazily and cache them
# ---------------------------------------------------------------------
from functools import lru_cache

from git import Repo
import yaml
from ..config import BUG_INDUCING_DIR, PYTHON_LIBS_DIR

# Assuming you have a dictionary mapping repo names to URLs
REPO_URL_MAP = {
    "pandas": "https://github.com/pandas-dev/pandas.git",
    "airflow": "https://github.com/apache/airflow.git",
    "numpy": "https://github.com/numpy/numpy",
    "ansible": "https://github.com/ansible/ansible.git",
    "sentry": "https://github.com/getsentry/sentry.git",
    "core": "https://github.com/home-assistant/core.git",
    "ray": "https://github.com/ray-project/ray.git",
}
# --- END CONFIG PLACEHOLDERS ---

BUG_INDUCING_COMMITS = {}


# @lru_cache(maxsize=None)
def _load_bug_inducing_commits(repo_name: str):
    """Load bug-inducing commits for a given repo based on filename."""

    yaml_files = list(BUG_INDUCING_DIR.glob(f"{repo_name}.*"))

    if not yaml_files:
        print(f"[WARN] No YAML found for repo: {repo_name}")
        # raise ValueError(f"No YAML found for repo: {repo_name}")
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


def get_registered_repos():
    return list(REPO_URL_MAP.keys())


def get_missing_repo_clones():
    """
    Returns a list of repository names that are registered in REPO_URL_MAP
    but do not have a local clone directory in PYTHON_LIBS_DIR.
    """
    missing_repos = []

    # Iterate over every registered repository name
    for repo_name in REPO_URL_MAP.keys():
        # Construct the full path where the clone is expected to be
        repo_path = PYTHON_LIBS_DIR / repo_name

        # Check if the path exists AND is a directory
        if not repo_path.is_dir():
            missing_repos.append(repo_name)

    return missing_repos


def download_missing_repo_clones():
    """
    Identifies missing repositories and clones them from their URLs
    in REPO_URL_MAP into the PYTHON_LIBS_DIR.
    """
    missing_repos = get_missing_repo_clones()

    if not missing_repos:
        print("[INFO] All registered repositories are already cloned locally.")
        return

    print(
        f"[INFO] Found {len(missing_repos)} repositories to clone: {', '.join(missing_repos)}"
    )

    # Ensure the destination directory exists
    if not PYTHON_LIBS_DIR.exists():
        print(f"[INFO] Creating base directory: {PYTHON_LIBS_DIR}")
        PYTHON_LIBS_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for repo_name in missing_repos:
        try:
            repo_url = REPO_URL_MAP[repo_name]
            clone_path = PYTHON_LIBS_DIR / repo_name

            print(f"  -> Cloning {repo_name} from {repo_url} to {clone_path}...")

            # Use git.Repo.clone_from to perform the cloning operation
            Repo.clone_from(url=repo_url, to_path=clone_path)

            print(f"  -> Successfully cloned {repo_name}.")
            success_count += 1

        except KeyError:
            print(f"[ERROR] Repository '{repo_name}' not found in REPO_URL_MAP.")
        except Exception as e:
            print(f"[ERROR] Failed to clone '{repo_name}'. Error: {e}")

    print(
        f"\n[SUMMARY] Successfully cloned {success_count} out of {len(missing_repos)} missing repositories."
    )
