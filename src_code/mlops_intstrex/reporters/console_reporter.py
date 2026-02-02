
from src_code.mlops_intstrex.reporters.progress_reporter import ProgressReporter


class ConsoleReporter(ProgressReporter):
    def start(self, total, description=""):
        self.total = total
        self.current = 0
        print(f"{description} started ({total} steps)")

    def advance(self, step=1):
        self.current += step
        print(f"{self.current}/{self.total}")

    def close(self):
        print("Finished")