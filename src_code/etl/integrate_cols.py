import argparse
import time
import pandas as pd
from tqdm import tqdm
from notebooks.logging_config import MyLogger
from src_code.config import ETL_MAPPINGS, INTERIM_DATA_DIR, LOG_DIR
from src_code.etl.features.textual_nlp import calculate_tfidf_features
from src_code.etl.utils import get_repo_instance
from src_code.versioning import VersionedFileManager

# ALL functions should take in a DataFrame and return an extended DataFrame
COLS_TO_INTEGRATE = {"tfifd_message": calculate_tfidf_features}

DF_TO_EXTEND = [
    ETL_MAPPINGS["train"]["current_newest"],
    ETL_MAPPINGS["test"]["current_newest"],
    ETL_MAPPINGS["validate"]["current_newest"],
]



def integrate_additional_columns(logger: MyLogger, subset: str = 'train', max_rows: int | None = None):
    db_file_versioner = VersionedFileManager(src_dir=INTERIM_DATA_DIR, file_name=subset + "_extended.feather")

    # for df_info in DF_TO_EXTEND:
    input_path = ETL_MAPPINGS[subset]['current_newest']
    # print(f"Integrating additional columns into {input_path}...")

    logger.log_check(f"Integrating additional columns into {input_path}...")
    df = pd.read_feather(input_path)

    if max_rows is not None:
        df = df.head(max_rows)

    for col_prefix, integration_func in COLS_TO_INTEGRATE.items():
        start = time.time()
        logger.log_check(f" - Integrating columns with prefix '{col_prefix}'...")
        df = integration_func(df, text_col="message", logger=logger)
        end = time.time()
        logger.log_result(f"   -> Completed in {end - start:.2f} seconds.")

    # output_path = input_path.parent / (input_path.stem + "_extended.feather")
    output_path = db_file_versioner.next_base_output

    logger.log_result(f" - Saving extended dataframe to {output_path}...")
    df.reset_index(drop=True).to_feather(output_path)


if __name__ == "__main__":
    argsparser = argparse.ArgumentParser(description="Integrate additional columns into ETL DataFrames.")
    argsparser.add_argument("--max-rows", type=int, required=False, default=None, help="Maximum number of rows to process from each DataFrame.")
    argsparser.add_argument("--subset", type=str, choices=["train", "test", "validate"], required=False, default='train', help="Subset to process (train, test, validate). If not specified, processes all.")
    
    args = argsparser.parse_args()


    SCRIPT_LOGGER = MyLogger(
        label="ETL",
        section_name="Column Integration",
        file_log_path=LOG_DIR / "etl_column_integration.log",
    )

    SCRIPT_LOGGER.log_check("Starting column integration process...")
    integrate_additional_columns(logger=SCRIPT_LOGGER, max_rows=args.max_rows, subset=args.subset)
    SCRIPT_LOGGER.log_result("Column integration process completed.")
