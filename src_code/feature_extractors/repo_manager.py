from pathlib import Path
import subprocess
from typing import Optional
import shutil

from ..config import PYTHON_LIBS_DIR

def ensure_repo_cloned(repo_name: str, repo_url: str) -> Path:
    """Clone repo into PYTHON_LIBS_DIR/<repo_name> if not already cloned."""
    repo_path = PYTHON_LIBS_DIR / repo_name

    if repo_path.exists():
        return repo_path

    print(f"[CLONE] {repo_name} → {repo_path}")
    subprocess.check_call(["git", "clone", repo_url, str(repo_path)])

    return repo_path


def checkout_commit(repo_path: Path, commit_hash: str):
    """Checkout a specific commit in the repo."""
    subprocess.check_call(["git", "-C", str(repo_path), "checkout", commit_hash])