import datetime
from git import Commit, Repo
import pandas as pd


# def calc_time_since_last_change(c: Commit) -> dict:
#     features = {"time_since_last_change": None}

#     if c.parents:
#         last_time = c.parents[0].committed_date
#         features["time_since_last_change"] = c.committed_date - last_time

#     return features

def calc_time_since_last_change(repo: Repo, c: Commit) -> dict:
    deltas = []

    if not c.parents:
        return {"time_since_last_change": 0}

    parent = c.parents[0]

    for diff in c.diff(parent):
        path = diff.b_path or diff.a_path
        if not path:
            continue

        try:
            history = list(repo.iter_commits(paths=path, max_count=2, rev=f"{c.hexsha}^"))
            if history:
                prev_commit = history[0]
                delta = c.committed_date - prev_commit.committed_date
                if delta >= 0:
                    deltas.append(delta)
        except Exception:
            continue

    if not deltas:
        return {"time_since_last_change": 0}

    return {
        "time_since_last_change": min(deltas)  # or np.mean(deltas)
    }

# def calc_recent_churn(c: Commit) -> dict:
#     features = {"recent_churn": 0}
    
#     if c.parents:
#         recent_loc_added = recent_loc_deleted = 0
#         parent = c.parents[0]
#         time_threshold = c.committed_date - 30 * 24 * 3600  # Last 30 days

#         for past_c in parent.repo.iter_commits(since=datetime.datetime.fromtimestamp(time_threshold)):
#             past_diff = past_c.diff(past_c.parents[0] if past_c.parents else None, create_patch=True)
#             for d in past_diff:
#                 patch = d.diff.decode(errors="ignore")
#                 recent_loc_added += patch.count('\n+') - patch.count('\n+++')
#                 recent_loc_deleted += patch.count('\n-') - patch.count('\n---')

#         features["recent_churn"] = recent_loc_added + recent_loc_deleted

#     return features
# def calc_recent_churn_from_df(df: pd.DataFrame, window_days: int = 30) -> pd.Series:
#     """
#     Calculates recent churn (sum of lines added + deleted in the last window_days) per author before each commit.
#     Uses only the DataFrame, no Git repo traversal.

#     Args:
#         df: DataFrame with 'repo', 'author_email', 'canonical_datetime', 'loc_added', 'loc_deleted'
#         window_days: lookback period

#     Returns:
#         pd.Series with recent churn for each commit
#     """
#     df_sorted = df.sort_values(by=['repo', 'author_email', 'canonical_datetime']).copy()
#     recent_churn_list = []

#     for (repo, author), group in df_sorted.groupby(['repo', 'author_email']):
#         timestamps = group['canonical_datetime'].tolist()
#         churn_values = (group['loc_added'] + group['loc_deleted']).tolist()

#         recent_churn = 0
#         window_start_idx = 0
#         window_td = pd.Timedelta(days=window_days)

#         for i, current_time in enumerate(timestamps):
#             # Remove old commits from the rolling window
#             while window_start_idx < i and (current_time - timestamps[window_start_idx]) > window_td:
#                 recent_churn -= churn_values[window_start_idx]
#                 window_start_idx += 1

#             # Store churn BEFORE adding current commit
#             recent_churn_list.append(recent_churn)

#             # Add current commit's churn to the running total
#             recent_churn += churn_values[i]

#     return pd.Series(recent_churn_list, index=df_sorted.index)

def calc_recent_churn_from_df(df: pd.DataFrame, window_days: int = 30) -> pd.Series:
    required = ["repo", "author_email", "canonical_datetime", "loc_added", "loc_deleted"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy()
    work_df["canonical_datetime"] = pd.to_datetime(work_df["canonical_datetime"], errors="coerce")

    if work_df["canonical_datetime"].isna().any():
        raise ValueError("canonical_datetime contains invalid datetime values")

    work_df = work_df.sort_values(
        by=["repo", "author_email", "canonical_datetime"]
    ).copy()

    work_df["commit_churn"] = work_df["loc_added"] + work_df["loc_deleted"]
    result = pd.Series(index=work_df.index, dtype="float64")

    window_td = pd.Timedelta(days=window_days)

    for (repo, author), group in work_df.groupby(["repo", "author_email"], sort=False):
        timestamps = group["canonical_datetime"].tolist()
        churn_values = group["commit_churn"].tolist()
        indices = group.index.tolist()

        running_sum = 0
        window_start = 0

        for i, current_time in enumerate(timestamps):
            while window_start < i and (current_time - timestamps[window_start]) > window_td:
                running_sum -= churn_values[window_start]
                window_start += 1

            result.loc[indices[i]] = running_sum
            running_sum += churn_values[i]

    return result.sort_index()