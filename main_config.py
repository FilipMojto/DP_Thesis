# --- Reproducibility ---
RANDOM_STATE = 42
TEST_SPLIT = 0.2

import numpy as np
import pandas as pd
import seaborn as sns

def setup():
    np.random.seed(RANDOM_STATE)
    pd.set_option('display.max_columns', 50)
    sns.set_theme(style="whitegrid", context="notebook", palette="muted")