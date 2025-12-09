LINE_TOKEN_FEATURES = ["todo", "fixme", "try", "except", "raise"]
BINARY_FEATURES = ["has_fix_kw", "has_bug_kw"]
TARGET = "label"
NUMERIC_FEATURES = [
    "author_exp_pre",
    "author_recent_activity_pre",
    "loc_added",
    "loc_deleted",
    "files_changed",
    "hunks_count",
    "msg_len",
    "ast_delta",
    "complexity_delta",
    "max_func_change",
    "time_since_last_change",
    "recent_churn",
]
EMBEDDINGS = ["code_embed", "msg_embed"]

ENGINEERED_FEATURES = [
    "loc_churn_ratio",
    "activity_per_exp",
    "loc_added_bucket",
    "extreme_churn_flag",
    "line_token_total",
]

ENGINEERED_FEATURES.extend([feature + "_ratio" for feature in LINE_TOKEN_FEATURES])

INTERACTION_FEATURES = ["loc_added", "loc_deleted", "hunks_count"]

for i in range(len(INTERACTION_FEATURES)):
    for j in range(i+1, len(INTERACTION_FEATURES)):
        f1 = INTERACTION_FEATURES[i]
        f2 = INTERACTION_FEATURES[j]
        ENGINEERED_FEATURES.append(f"{f1}_x_{f2}")