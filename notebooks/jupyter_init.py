# src_code/jupyter_init.py
import sys
from pathlib import Path

def setup():
    # ROOT = Path(__file__).resolve().parents[1]  # project root
    # if str(ROOT) not in sys.path:
    #     sys.path.append(str(ROOT))
    # Get the directory of the notebook, then go up one level to the project root
    NOTEBOOK_DIR = Path().resolve()
    PROJECT_ROOT = NOTEBOOK_DIR.parent
    # Add the project root to the system path
    sys.path.append(str(PROJECT_ROOT))