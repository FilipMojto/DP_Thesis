from typing import Dict

from git import Commit


def calculate_change_churn_metrics(diff_text: Commit) -> Dict[str, int]:
    """Calculate change churn metrics for a given commit.
    Args:
        c (Commit): GitPython Commit object.
    Returns:
        Dict[str, int]: Dictionary with change churn metrics.
    """

    # diff_text = c.diff(c.parents[0] if c.parents else None, create_patch=True)
    loc_added = loc_deleted = files_changed = hunks_count = 0

    for d in diff_text:
        if d.a_blob and d.b_blob:
            files_changed += 1
            patch = d.diff.decode(errors="ignore")
            loc_added += patch.count('\n+') - patch.count('\n+++')
            loc_deleted += patch.count('\n-') - patch.count('\n---')
            hunks_count += patch.count('@@')
    
    return {
        "loc_added": loc_added,
        "loc_deleted": loc_deleted,
        "files_changed": files_changed,
        "hunks_count": hunks_count
    }