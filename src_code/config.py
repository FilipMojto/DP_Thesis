__ALL__ = [
    "DATA_DIR",
    "DEFECTORS_DIR",
    "JIT_FILE",
    "BUG_INDUCING_DIR",
    "PREPROCESSED_DATA_DIR",
]

from pathlib import Path
from typing import Literal

from src_code.versioning import find_newest_version, next_version_path

SubsetType = Literal['train', 'test', 'validate']


# 1. Get the directory of the current file (config.py or similar)
# Use .resolve() to get the absolute path, and .parent to get the directory
CONFIG_FILE_DIR = Path(__file__).resolve()

# 2. Define the Project Root (assuming 'data' is two levels up)
# Adjust the number of .parent calls based on your actual file structure:
# If this file is in 'code/config.py' and 'data' is in the root, it's two parents up.
PROJECT_ROOT = CONFIG_FILE_DIR.parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
RANDOM_FOREST_MODEL = MODEL_DIR / "random_forest_pipeline.joblib"

# 3. Define all data paths relative to the Project Root
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"


DEFECTORS_DIR = RAW_DATA_DIR / "defectors"
# INTERIM_DATA_DIR = DATA_DIR / "extracted"
# PROCESSED_DATA_DIR = DATA_DIR / "preprocessed"

BUG_INDUCING_DIR = RAW_DATA_DIR / "bug_inducing_commits"
PYTHON_LIBS_DIR = EXTERNAL_DATA_DIR / "python_libs"

JIT_DIR = DEFECTORS_DIR / "jit_bug_prediction_splits/time"

JIT_TRAIN_FEATHER_FILE = JIT_DIR / "train.feather"
JIT_TEST_FEATHER_FILE = JIT_DIR / "test.feather"

# JIT_FEATHER_FILE = EXTRACTED_DATA_DIR / "jit_bug_prediction_splits/time/train.feather"
EXTRATED_TRAIN_DF_FILE = INTERIM_DATA_DIR / "train_labeled_features_partial.feather"
EXTRACTED_TEST_DF_FILE = INTERIM_DATA_DIR / "test_labeled_features_partial.feather"

PREPROCESSED_TRAIN_DF_FILE = PROCESSED_DATA_DIR / "train_preprocessed.feather"
PREPROCESSED_TEST_DF_FILE = PROCESSED_DATA_DIR / "test_preprocessed.feather" 

ENGINEERED_TRAIN_DF_FILE = PROCESSED_DATA_DIR / "train_engineered.feather"
ENGINEERED_TEST_DF_FILE = PROCESSED_DATA_DIR / "test_engineered.feather" 

FITTED_TRANSFORMER = MODEL_DIR / 'fitted_preprocessor.joblib'
ENGINEERING_TRANSFORMER = MODEL_DIR / 'engineering_preprocessor.joblib'

def get_output(input_file: Path):
    return INTERIM_DATA_DIR / (input_file.stem + "_labeled_features_partial.feather")

# def get_copy(out_file: Path):
#     return INTERIM_DATA_DIR / (out_file.stem + "_copy.feather")


ETL_MAPPINGS = {
    "train": {
        "input": JIT_TRAIN_FEATHER_FILE,
        # "output": EXTRACTED_DATA_DIR
        # / (JIT_TRAIN_FEATHER_FILE.stem + "_labeled_features_partial.feather"),
        "base_output": get_output(JIT_TRAIN_FEATHER_FILE)
    },
    "test": {
        "input": JIT_TEST_FEATHER_FILE,
        "base_output": get_output(JIT_TEST_FEATHER_FILE)
    }
}

# for mapping in EXTRACTION_MAPPINGS.values():
#     mapping['copy'] = get_copy(mapping['output'])

for mapping in ETL_MAPPINGS.values():
    base_output = mapping["base_output"]

    newest_path, newest_version = find_newest_version(base_output)
    next_path = next_version_path(base_output)

    # mapping["base_output"] = base_output              # stable logical name
    mapping["current_newest"] = newest_path           # Path | None
    mapping["current_newest_version"] = newest_version
    mapping["next_output"] = next_path                # where to write next


PREPROCESSING_MAPPINGS = {
    "train": {
        "input": ETL_MAPPINGS['train']['current_newest'],
        "output": PREPROCESSED_TRAIN_DF_FILE
    },
    "test": {
        "input": ETL_MAPPINGS['test']['current_newest'],
        "output": PREPROCESSED_TEST_DF_FILE
    }
}

ENGINEERING_MAPPINGS = {
    "train": {
        "input": PREPROCESSING_MAPPINGS['train']['output'],
        "output": ENGINEERED_TRAIN_DF_FILE
    },
    "test": {
        "input": PREPROCESSING_MAPPINGS["test"]['output'],
        "output": ENGINEERED_TEST_DF_FILE
    }
}