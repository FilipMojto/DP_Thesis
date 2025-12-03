

import datetime
from git import Commit
import pandas as pd


def calc_time_since_last_change(c: Commit) -> dict:
    features = {"time_since_last_change": None}

    if c.parents:
        last_time = c.parents[0].committed_date
        features["time_since_last_change"] = c.committed_date - last_time

    return features

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
def calc_recent_churn_from_df(df: pd.DataFrame, window_days: int = 30) -> pd.Series:
    """
    Calculates recent churn (sum of lines added + deleted in the last window_days) per author before each commit.
    Uses only the DataFrame, no Git repo traversal.

    Args:
        df: DataFrame with 'repo', 'author_email', 'canonical_datetime', 'loc_added', 'loc_deleted'
        window_days: lookback period

    Returns:
        pd.Series with recent churn for each commit
    """
    df_sorted = df.sort_values(by=['repo', 'author_email', 'canonical_datetime']).copy()
    recent_churn_list = []

    for (repo, author), group in df_sorted.groupby(['repo', 'author_email']):
        timestamps = group['canonical_datetime'].tolist()
        churn_values = (group['loc_added'] + group['loc_deleted']).tolist()

        recent_churn = 0
        window_start_idx = 0
        window_td = pd.Timedelta(days=window_days)

        for i, current_time in enumerate(timestamps):
            # Remove old commits from the rolling window
            while window_start_idx < i and (current_time - timestamps[window_start_idx]) > window_td:
                recent_churn -= churn_values[window_start_idx]
                window_start_idx += 1

            # Store churn BEFORE adding current commit
            recent_churn_list.append(recent_churn)

            # Add current commit's churn to the running total
            recent_churn += churn_values[i]

    return pd.Series(recent_churn_list, index=df_sorted.index)