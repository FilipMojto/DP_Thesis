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
    "recent_churn"
]
