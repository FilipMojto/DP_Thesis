import re
from pathlib import Path
from typing import Optional, Tuple


VERSION_RE = re.compile(r"_v(\d+)$")


def extract_version(path: Path) -> Optional[int]:
    """
    Returns the integer version from a filename suffix '_vX', or None.
    """
    m = VERSION_RE.search(path.stem)
    return int(m.group(1)) if m else None


def find_newest_version(base_output: Path) -> Tuple[Optional[Path], int]:
    """
    Finds the newest versioned file for a given base output path.

    Returns:
        (path_to_newest_version or None, newest_version_number)
    """
    parent = base_output.parent
    base_stem = base_output.stem

    candidates = parent.glob(f"{base_stem}_v*.feather")

    newest_path = None
    newest_version = 0

    for p in candidates:
        v = extract_version(p)
        if v is not None and v > newest_version:
            newest_version = v
            newest_path = p

    if newest_path is None:
        newest_path = Path(f"{base_stem}_v1.feather")
        newest_version = 1

    # print(f"Newest version for {base_output} is v{newest_version} at {newest_path}")
    return newest_path, newest_version


def next_version_path(base_output: Path) -> Path:
    _, newest_version = find_newest_version(base_output)
    next_v = newest_version + 1
    return base_output.with_name(f"{base_output.stem}_v{next_v}{base_output.suffix}")