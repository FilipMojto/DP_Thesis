__ALL__ = [
    "DATA_DIR",
    "DEFECTORS_DIR",
    "JIT_FILE",
    "BUG_INDUCING_DIR",
    "PREPROCESSED_DATA_DIR",
]

from pathlib import Path


# 1. Get the directory of the current file (config.py or similar)
# Use .resolve() to get the absolute path, and .parent to get the directory
CONFIG_FILE_DIR = Path(__file__).resolve()

# 2. Define the Project Root (assuming 'data' is two levels up)
# Adjust the number of .parent calls based on your actual file structure:
# If this file is in 'code/config.py' and 'data' is in the root, it's two parents up.
PROJECT_ROOT = CONFIG_FILE_DIR.parent.parent

# 3. Define all data paths relative to the Project Root
DATA_DIR = PROJECT_ROOT / "data"
DEFECTORS_DIR = DATA_DIR / "defectors"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"

BUG_INDUCING_DIR = DATA_DIR / "bug_inducing_commits"
PYTHON_LIBS_DIR = DATA_DIR / "python_libs"

JIT_DIR = DEFECTORS_DIR / "jit_bug_prediction_splits/time"

JIT_TRAIN_FEATHER_FILE = JIT_DIR / "train.feather"
JIT_TEST_FEATHER_FILE = JIT_DIR / "test.feather"

# JIT_FEATHER_FILE = EXTRACTED_DATA_DIR / "jit_bug_prediction_splits/time/train.feather"
EXTRATED_TRAIN_DF_FILE = EXTRACTED_DATA_DIR / "train_labeled_features_partial.feather"

def get_output(input_file: Path):
    return EXTRACTED_DATA_DIR / (input_file.stem + "_labeled_features_partial.feather")

def get_copy(out_file: Path):
    return EXTRACTED_DATA_DIR / (out_file.stem + "_copy.feather")


EXTRACTION_MAPPINGS = {
    "train": {
        "input": JIT_TRAIN_FEATHER_FILE,
        # "output": EXTRACTED_DATA_DIR
        # / (JIT_TRAIN_FEATHER_FILE.stem + "_labeled_features_partial.feather"),
        "output": get_output(JIT_TRAIN_FEATHER_FILE)
    },
    "test": {
        "input": JIT_TEST_FEATHER_FILE,
        "output": get_output(JIT_TEST_FEATHER_FILE)
    }
}

for mapping in EXTRACTION_MAPPINGS.values():
    mapping['copy'] = get_copy(mapping['output'])