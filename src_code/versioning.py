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

    candidates = parent.glob(f"{base_stem}_v*{extension}")

    newest_path = None
    newest_version = 0

    for p in candidates:
        v = extract_version(p)
        if v is not None and v > newest_version:
            newest_version = v
            newest_path = p

    if newest_path is None:
        newest_path = Path(f"{base_stem}_v1{extension}")
        newest_version = 1

    # print(f"Newest version for {base_output} is v{newest_version} at {newest_path}")
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

    # def __init__(self, src_dir: Path, file_name: str, extension: str):
    def __init__(self, file_path: Path):
        # self.src_dir = src_dir
        # self.file_name = file_name
        self.file_path = file_path
        self.extension = file_path.suffix
        # self.base_output = src_dir / file_name
        # self.base_output = file_path.parent / file_path.stem

        # self.current_newest, self.current_newest_version = find_newest_version(self.base_output)
        self.update()

    def update(self):
        """
        Refreshes the current newest version and path.
        """
        self.current_newest, self.current_newest_version = find_newest_version(
            self.file_path, extension=self.extension
        )
        # self.next_base_output = next_version_path(self.base_output)
        self.next_base_output = self.file_path.with_name(
            f"{self.file_path.stem}_v{self.current_newest_version + 1}{self.file_path.suffix}"
        )

    # def update_to_next_version(self):
    #     self.update()
