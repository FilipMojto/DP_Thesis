from typing import Protocol, Any

from src_code.mlops_intstrex.reporters.progress_reporter import ProgressReporter


class ProgressAdapter(Protocol):
    """The standard interface for any progress-tracked execution."""
    def execute(self, reporter: ProgressReporter, *args, **kwargs) -> Any:
        ...