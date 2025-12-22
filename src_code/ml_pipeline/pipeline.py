from notebooks.constants import NUMERIC_FEATURES
from .preprocessing import drop_invalid_rows
from .df_load import load_df
from ..config import PREPROCESSING_MAPPINGS, SubsetType

if __name__ == "__main__":
    subset: SubsetType = "train"
    target_df_path = TARGET_DF_FILE = PREPROCESSING_MAPPINGS[subset]["input"]

    print("HERE")
    target_df = load_df(target_df_path)
    drop_invalid_rows(
        df=target_df,
        numeric_features=NUMERIC_FEATURES,
        # row_filters={"time_since_last_change": target_df["time_since_last_change"] < 0},
        row_filters={"time_since_last_change": lambda s: s>= 0}
    )
