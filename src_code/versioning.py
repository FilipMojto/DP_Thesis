import re
from pathlib import Path
from typing import Optional, Tuple

from notebooks.logging_config import MyLogger


VERSION_RE = re.compile(r"_v(\d+)$")


def extract_version(path: Path) -> Optional[int]:
    """
    Returns the integer version from a filename suffix '_vX', or None.
    """
    m = VERSION_RE.search(path.stem)
    return int(m.group(1)) if m else None


def find_newest_version(
    base_output: Path, extension: str = ".feather"
) -> Tuple[Optional[Path], int]:
    """
    Finds the newest versioned file for a given base output path.

    Returns:
        (path_to_newest_version or None, newest_version_number)
    """
    parent = base_output.parent
    base_stem = base_output.stem

    clean_stem = re.sub(r'_v\d+$', '', base_stem)
    candidates = parent.glob(f"{clean_stem}_v*{extension}")

    newest_path = None
    newest_version = 0

    for p in candidates:
        v = extract_version(p)
        if v is not None and v > newest_version:
            newest_version = v
            newest_path = p

    return newest_path, newest_version


def next_version_path(base_output: Path) -> Path:
    _, newest_version = find_newest_version(base_output)
    next_v = newest_version + 1
    return base_output.with_name(f"{base_output.stem}_v{next_v}{base_output.suffix}")


class VersionedFileManager:
    """
    Manages versioned files for a given base output path.
    Provides methods to get the current newest version and the next version path.

    Constraints:
    - Versioned files must follow the naming convention '_vX' where X is an integer.
    - Only one versioned file can exist per base name.
    - Limited to a single directory and file per instance.

    Limitations:
    - Assumes versioning is done via '_vX' suffix in filenames.
    - Synchronous; does not handle concurrent updates.
    - Does not create or write files; only manages paths.
    """

    def __init__(self, file_path: Path, logger: MyLogger, throw_not_found_err: bool = False):
        self.file_path = file_path
        self.extension = file_path.suffix
        self.logger = logger
  
        self.update()
        self.logger.log_result(f"Current newest version: {self.current_newest.absolute() if self.current_newest else self.current_newest}")

        if throw_not_found_err and self.current_newest == None:
            raise FileNotFoundError("File not found!")
        
    def update(self):
        """
        Refreshes the current newest version and path.
        """
        self.current_newest, self.current_newest_version = find_newest_version(
            self.file_path, extension=self.extension
        )

        self.next_base_output = self.file_path.with_name(
            f"{self.file_path.stem}_v{self.current_newest_version + 1}{self.file_path.suffix}"
        )
