import subprocess
from pathlib import Path

def get_diff(repo_path: Path, commit_hash: str) -> str:
    cmd = ["git", "-C", str(repo_path), "show", commit_hash, "--unified=0"]
    return subprocess.check_output(cmd, text=True, errors="ignore")


def extract_diff_features(diff_text: str) -> dict:
    added = deleted = files = hunks = 0

    for line in diff_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            continue

        if line.startswith("diff --git"):
            files += 1
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1

    return {
        "loc_added": added,
        "loc_deleted": deleted,
        "files_changed": files,
        "hunks": hunks,
    }