from notebooks.constants import NUMERIC_FEATURES
from .preprocessing import drop_invalid_rows, transform
from .df_load import load_df
from ..config import PREPROCESSING_MAPPINGS, SubsetType

RANDOM_STATE = 42

if __name__ == "__main__":
    subset: SubsetType = "train"
    target_df_path = TARGET_DF_FILE = PREPROCESSING_MAPPINGS[subset]["input"]


    print("HERE")
    target_df = load_df(target_df_path)
    target_df = drop_invalid_rows(
        df=target_df,
        # numeric_features=NUMERIC_FEATURES,
        # row_filters={"time_since_last_change": target_df["time_since_last_change"] < 0},
        row_filters={"time_since_last_change": lambda s: s>= 0}
    )

    target_df, fitted_transformer = transform(df=target_df,
                   subset=subset,
                   random_state=RANDOM_STATE,
                   )
    
    print(
        f"Code embeddings explain "
        f"{fitted_transformer.pca_explained_variance('code_embed'):.2%} of variance"
    )

    print(
        f"Message embeddings explain "
        f"{fitted_transformer.pca_explained_variance('msg_embed'):.2%} of variance"
    )
