def extract_dev_features(df_history, row):
    author = row.author
    ts = row.timestamp

    author_df = df_history[df_history["author"] == author]

    return {
        "author_exp": len(author_df),
        "author_recent_activity": len(author_df[author_df.timestamp > ts - 86400*30]),
    }