__ALL__ = ['DERIVED_FEATURES', 'BUCKET_MAPPINGS']

import numpy as np



DERIVED_FEATURES = {
    "loc_churn_ratio": lambda df: df["loc_added"] / (df["loc_deleted"] + 1),
    "activity_per_exp": lambda df: df["author_recent_activity_pre"]
    / (df["author_exp_pre"] + 1),
}

BUCKET_MAPPINGS = {
    "loc_added": {
        (0, 3): "very_small",
        (3, 6): "small",
        (6, 10): "medium",
        (10, 20): "large",
        (20, np.inf): "very_large",
    },
    "author_exp_pre": {
        (0, 1): "junior",
        (1, 5): "mid",
        (5, np.inf): "senior",
    },
}

DROP_COLS = [
    "commit",
    "repo",
    "filepath",
    "author_email",
    "datetime",
    "canonical_datetime",
    "content",
    "methods",
    "lines"
    # "files_changed",
    # "loc_added_bucket"
]