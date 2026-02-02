from tqdm import tqdm

from src_code.mlops_intstrex.reporters.progress_reporter import ProgressReporter



# class TqdmReporter(ProgressReporter):
#     def __init__(self):
#         self.bar = None

#     def start(self, total: int, description: str = ""):
#         self.bar = tqdm(total=total, desc=description)

#     def advance(self, step: int = 1):
#         self.bar.update(step)

#     def close(self):
#         self.bar.close()
class TqdmReporter(ProgressReporter):
    def __init__(self):
        self.bar = None

    def start(self, total: int, description: str = ""):
        self.bar = tqdm(total=total, desc=description)

    def advance(self, step: int = 1):
        if self.bar is not None:
            self.bar.update(step)

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    # This prevents the __del__ crash in worker processes
    def __getstate__(self):
        # When pickling for a worker, don't send the bar object
        state = self.__dict__.copy()
        state['bar'] = None 
        return state