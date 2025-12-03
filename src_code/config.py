__ALL__ = ["DATA_DIR", "DEFECTORS_DIR", "JIT_FILE", "BUG_INDUCING_DIR"]

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
JIT_FILE = DEFECTORS_DIR / "jit_bug_prediction_splits/time/train.feather"
BUG_INDUCING_DIR = DATA_DIR / "bug_inducing_commits"
PYTHON_LIBS_DIR = DATA_DIR / "python_libs"