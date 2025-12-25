import pandas as pd
import seaborn as sns

def setup():
    pd.set_option('display.max_columns', 50)
    sns.set_theme(style="whitegrid", context="notebook", palette="muted")