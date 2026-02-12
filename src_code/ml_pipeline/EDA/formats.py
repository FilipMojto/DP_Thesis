

from pathlib import Path

from matplotlib import pyplot as plt


def save_as_pdf(path: Path):
    plt.savefig(path, bbox_inches="tight")        # Vector for LaTeX


def save_as_png(path: Path):
    plt.savefig(path, bbox_inches="tight", dpi=300)  # High-res for Word
