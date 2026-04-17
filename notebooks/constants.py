# from src_code.config import LOG_DIR


from pathlib import Path


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
    # "loc_added_bucket",
    # "extreme_churn_flag",
    "line_token_total",
]

ENGINEERED_FEATURES.extend([feature + "_ratio" for feature in LINE_TOKEN_FEATURES])

INTERACTION_FEATURES = ["loc_added", "loc_deleted", "hunks_count"]

for i in range(len(INTERACTION_FEATURES)):
    for j in range(i+1, len(INTERACTION_FEATURES)):
        f1 = INTERACTION_FEATURES[i]
        f2 = INTERACTION_FEATURES[j]
        ENGINEERED_FEATURES.append(f"{f1}_x_{f2}")

LOG_FILE = Path().resolve().parent / "notebooks.log"
LOG_DIR = Path().resolve().parent / "notebooks/logs"

CHURN_METRICS = ['loc_added', 'loc_deleted', 'files_changed', 'hunks_count']
TEXTUAL_METRICS = ['msg_len', 'has_fix_kw', 'has_bug_kw']
DEVELOPER_METRICS = ['author_exp_pre', 'author_recent_activity_pre']
HISTORIC_TEMPORAL_METRICS = ['recent_churn', 'time_since_last_change']


LINE_LEVEL_SPECIFIC = ['line_context_embed', 'line_token_features']
STRUCTURAL_METRICS = ['ast_delta', 'max_func_change', 'complexity_delta']

STATISTICAL_METRICS = CHURN_METRICS + TEXTUAL_METRICS + DEVELOPER_METRICS + HISTORIC_TEMPORAL_METRICS
# NOTE: Also includes embeddings and tfidf
NON_STATISTICAL_METRICS = LINE_TOKEN_FEATURES


# CODE_SUTRUCTURAL_METRICS = ['ast_node_delta', 'max_func_change_size', 'complexity_delta']
