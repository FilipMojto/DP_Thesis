from collections import defaultdict
import datetime
import git
import pandas as pd


def pre_calculate_author_metrics(df: pd.DataFrame, get_repo_func) -> pd.DataFrame:
    """
    Calculates author experience and recent activity incrementally, fetching missing 
    author email and datetime from the local Git repository first.

    pre_calculate_author_metrics computes author-level historical features for each commit:

    1. Author experience (author_exp_pre)
    → How many commits this author had made in the same repository before this commit

    2. Recent activity (author_recent_activity_pre)
    → How many commits this author made in the last 30 days before this commit

    These are time-dependent features and must be computed incrementally in chronological order.
        
    Args:
        df: The input DataFrame containing 'repo', 'commit'.
        get_repo_func: A function (repo_name: str) -> git.Repo instance.
            
    Returns:
        The DataFrame with 'author_email', 'canonical_datetime', 'author_exp_pre', 
        and 'author_recent_activity_pre' columns added.
    """
    print("[PRECALC] Fetching author metadata from Git...")
    
    # 1. METADATA FETCH PASS
    metadata = []
    
    # Cache Git Repo objects
    repo_cache = {} 

    # Why this step exists:
    # Your input dataframe does not contain author identity or commit time
    # These are not derivable from ML data alone
    # They must be fetched from Git (expensive I/O)
    for row in df[['repo', 'commit']].itertuples(index=False):
        repo_name = row.repo
        commit_hash = row.commit
        
        if repo_name not in repo_cache:
            try:
                repo_cache[repo_name] = get_repo_func(repo_name)
            except Exception as e:
                print(f"[ERROR] Could not load repo {repo_name}: {e}. Skipping commit.")
                metadata.append((None, None, None))
                continue
                
        repo = repo_cache[repo_name]

        try:
            c = repo.commit(commit_hash)
            # Ensure datetime is timezone-aware pandas/datetime object for calculations
            dt = datetime.datetime.fromtimestamp(c.committed_date, tz=datetime.timezone.utc)
            email = c.author.email
            
        except git.BadObject:
            print(f"[WARN] Commit {commit_hash} not found in {repo_name}. Skipping.")
            email = None
            dt = None
            
        metadata.append((repo_name, commit_hash, email, dt))

    # Convert metadata list to a temporary DF and merge it back for cleanup
    meta_df = pd.DataFrame(metadata, columns=['repo_meta', 'commit_meta', 'author_email', 'canonical_datetime'])
    
    # Merge on the index or simply replace the columns if rows match 1:1 (assuming no reordering yet)
    df = df.reset_index(drop=True).copy()
    df['author_email'] = meta_df['author_email']
    df['canonical_datetime'] = meta_df['canonical_datetime']
    
    # Filter out rows where metadata could not be fetched (commit not found)
    df = df.dropna(subset=['author_email', 'canonical_datetime'])
    print(f"[PRECALC] Finished metadata fetch. {len(df)} valid commits remaining.")

    # 2. CALCULATION PASS
    
    # 1. Ensure the DataFrame is sorted chronologically by repo and then time
    # This is critical for incremental calculation.
    df_sorted = df.sort_values(by=['repo', 'canonical_datetime']).reset_index(drop=True)
    
    # Cache structure: {repo: {author_email: total_commits}}
    author_repo_commits = defaultdict(lambda: defaultdict(int))
    
    # Cache structure for recent activity: {author_email: list of commit_datetimes}
    author_recent_activity_cache = defaultdict(list)
    RECENT_ACTIVITY_DAYS = 30 
    
    exp_list = []
    recent_activity_list = []
    
    for _, row in df_sorted.iterrows():
        repo = row['repo']
        # Use the newly fetched email
        author_email = row['author_email'] 
        current_time = row['canonical_datetime']
        
        # --- Author Experience (Total Commits) ---
        experience_before = author_repo_commits[repo][author_email]
        exp_list.append(experience_before)
        author_repo_commits[repo][author_email] += 1
        
        # --- Recent Activity (30-day window) ---
        author_recent_activity_cache[author_email].append(current_time)
        cutoff_time = current_time - pd.Timedelta(days=RECENT_ACTIVITY_DAYS)
        
        recent_commits = author_recent_activity_cache[author_email]
        
        # Filter in place to remove old commits
        # Note: Filtering a list repeatedly is O(N) where N is list size, better than O(N*C) Git call.
        recent_commits[:] = [t for t in recent_commits if t >= cutoff_time]
        
        # The count *before* this commit is the size of the list minus one
        activity_before = max(0, len(recent_commits) - 1)
        recent_activity_list.append(activity_before)
        
    # Append the new columns and return
    df_sorted['author_exp_pre'] = exp_list
    df_sorted['author_recent_activity_pre'] = recent_activity_list
    
    return df_sorted