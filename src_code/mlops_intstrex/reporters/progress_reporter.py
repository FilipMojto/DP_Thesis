# from abc import ABC, abstractmethod

# class ProgressReporter(ABC):

#     @abstractmethod
#     def start(self, total: int, description: str = ""):
#         pass

#     @abstractmethod
#     def advance(self, step: int = 1):
#         pass

#     @abstractmethod
#     def close(self):
#         pass
from typing import Protocol


class ProgressReporter(Protocol):
    def start(self, total: int, description: str = "") -> None: ...
    def advance(self, step: int = 1) -> None: ...
    def close(self) -> None: ...