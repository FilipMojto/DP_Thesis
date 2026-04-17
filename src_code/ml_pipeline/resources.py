


import os
import psutil
from dataclasses import dataclass, field
from typing import Literal

CoreModeType = Literal['manual', 'all']

DEF_RESERVE_CORES = 2
# DEF_NUM_OF_CORES = psutil.cpu_count(logical=False) - DEF_RESERVE_CORES
DEF_NUM_OF_CORES = 1
DEF_CORE_MODE_TYPE: CoreModeType = 'manual'
def get_n_jobs(reserve: int = 2) -> int:
    """
    Return number of cores to use, reserving some for system responsiveness.
    """
    total_cores = (os.cpu_count() / 2) or 1
    return max(1, total_cores - reserve)

@dataclass
class CoreConfig:
    reserve_cores: int = DEF_RESERVE_CORES
    num_of_cores: int = DEF_NUM_OF_CORES
    mode: CoreModeType = DEF_CORE_MODE_TYPE
    
    # Internal field to store the actual calculated value
    _final_n_jobs: int = field(init=False, repr=False)

    def __post_init__(self):
        self._calculate_cores()

    # def _calculate_cores(self):
    #     # 1. Determine total physical availability
    #     # We use logical=False if you want physical cores, 
    #     # but for Grid Search, logical cores (Hyperthreading) are usually fine.
    #     total_available = os.cpu_count() or 1
        
    #     # Rule (a): Reserve cores must be complied with
    #     # We ensure we don't try to reserve more cores than exist
    #     actual_reserves = min(self.reserve_cores, total_available - 1)
    #     max_allowed = max(1, total_available - actual_reserves)

    #     if self.mode == 'all':
    #         # Rule (c) variant: If 'all', use everything except reserves
    #         self._final_n_jobs = max_allowed
    #     else:
    #         # Rule (b): Manual mode
    #         # Comply with manual setting only if it doesn't exceed the (Total - Reserved) limit
    #         if self.num_of_cores > max_allowed:
    #             self._final_n_jobs = max_allowed
    #         else:
    #             self._final_n_jobs = max(1, self.num_of_cores)
    def _calculate_cores(self):
        total_available = os.cpu_count() or 1
        actual_reserves = max(0, min(self.reserve_cores, total_available - 1))
        max_allowed = max(1, total_available - actual_reserves)

        if self.mode == "all":
            self._final_n_jobs = max_allowed
        else:
            self._final_n_jobs = max(1, min(self.num_of_cores, max_allowed))

    @property
    def n_jobs(self) -> int:
        """This is what you pass to GridSearchCV(n_jobs=...)"""
        return self._final_n_jobs

    def __str__(self) -> str:
        total = os.cpu_count()
        return (
            f"--- Core Allocation Report ---\n"
            f"Mode:            {self.mode.upper()}\n"
            f"System Total:    {total} cores\n"
            f"Reserved:        {self.reserve_cores} cores\n"
            f"Target Manual:   {self.num_of_cores if self.mode == 'manual' else 'N/A'}\n"
            f"Final Allocated: {self._final_n_jobs} cores\n"
            f"------------------------------"
        )
