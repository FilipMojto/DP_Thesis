__ALL__ = [
    "DATA_DIR",
    "DEFECTORS_DIR",
    "JIT_FILE",
    "BUG_INDUCING_DIR",
    "PREPROCESSED_DATA_DIR",
]

from pathlib import Path
from typing import Literal

SubsetType = Literal['train', 'test', 'val']
SupportedModel = Literal['RF', 'XGB', 'NN', "LR", "DUMMY_STRATIFIED", "DUMMY_MOST_FREQUENT", "ENSEMBLE_VOTING", "ENSEMBLE_STACKING"]


CONFIG_FILE_DIR = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_FILE_DIR.parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
TUNED_DIR = MODEL_DIR / "tuned_params"
PIPELINE_DIR = MODEL_DIR / "pipelines"
LOG_DIR = PROJECT_ROOT / "logs"
EXPERIMENT_DIR = LOG_DIR / "experiments"
SRC_DIR = PROJECT_ROOT / "src_code"
TESTING_DIR = PROJECT_ROOT / "testing"
REPORTS_DIR = PROJECT_ROOT / "reports"

ML_PIPELINE_DIR = SRC_DIR / "ml_pipeline"
EVALUATION_DIR = REPORTS_DIR / "evaluation"
EDA_DIR = REPORTS_DIR / "eda"

RANDOM_FOREST_MODEL = MODEL_DIR / "random_forest_pipeline.joblib"

CACHE_DIR = PROJECT_ROOT / 'cache'

# 3. Define all data paths relative to the Project Root
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTRACTED_DFS = INTERIM_DATA_DIR / "etl2"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

EXTENDED_DATA_DIR = INTERIM_DATA_DIR / "extended"
RELABELED_DATA_DIR = INTERIM_DATA_DIR / "relabelled"
ENGINEERED_DATA_DIR = PROCESSED_DATA_DIR / "engineered"
TRANSFORMED_DATA_DIR = PROCESSED_DATA_DIR / "transformed"


DEFECTORS_DIR = RAW_DATA_DIR / "defectors"

BUG_INDUCING_DIR = RAW_DATA_DIR / "bug_inducing_commits"
PYTHON_LIBS_DIR = EXTERNAL_DATA_DIR / "python_libs"

JIT_DIR = DEFECTORS_DIR / "jit_bug_prediction_splits/time"

JIT_TRAIN_FEATHER_FILE = JIT_DIR / "train.feather"
JIT_TEST_FEATHER_FILE = JIT_DIR / "test.feather"
JIT_VAL_FEATHER_FILE = JIT_DIR / "val.feather"

EXTRATED_TRAIN_DF_FILE = INTERIM_DATA_DIR / "train_labeled_features_partial.feather"
EXTRACTED_TEST_DF_FILE = INTERIM_DATA_DIR / "test_labeled_features_partial.feather"

PREPROCESSED_TRAIN_DF_FILE = PROCESSED_DATA_DIR / "train_preprocessed.feather"
PREPROCESSED_TEST_DF_FILE = PROCESSED_DATA_DIR / "test_preprocessed.feather" 

ENGINEERED_TRAIN_DF_FILE = PROCESSED_DATA_DIR / "train_engineered.feather"
ENGINEERED_TEST_DF_FILE = PROCESSED_DATA_DIR / "test_engineered.feather" 
ENGINEERED_VAL_DF_FILE = PROCESSED_DATA_DIR / "val_engineered.feather"

FITTED_TRANSFORMER = PIPELINE_DIR / 'fitted_preprocessor.joblib'
ENGINEERING_TRANSFORMER = MODEL_DIR / 'engineering_preprocessor.joblib'

def get_output(input_file: Path):
    return INTERIM_DATA_DIR / (input_file.stem + "_labeled_features_partial.feather")


ETL_PATH_MAPPINGS = {

    "train": {
        "input": JIT_TRAIN_FEATHER_FILE,
        # "output": EXTRACTED_DATA_DIR
        # / (JIT_TRAIN_FEATHER_FILE.stem + "_labeled_features_partial.feather"),
        "base_output": get_output(JIT_TRAIN_FEATHER_FILE)
    },
    "test": {
        "input": JIT_TEST_FEATHER_FILE,
        "base_output": get_output(JIT_TEST_FEATHER_FILE)
    },
    "val": {
        "input": JIT_VAL_FEATHER_FILE,
        "base_output": get_output(JIT_VAL_FEATHER_FILE)
    }

}

