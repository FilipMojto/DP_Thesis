from tqdm import tqdm



class TqdmReporter:
    def __init__(self):
        self.bar = None

    def start(self, total: int, description: str = ""):
        self.bar = tqdm(total=total, desc=description)

    def advance(self, step: int = 1):
        self.bar.update(step)

    def close(self):
        self.bar.close()