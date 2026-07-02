import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from logger import get_logger

log = get_logger(__name__)

mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def save_all(prefix: str, fmt: str = "pdf") -> None:
    """Save every open figure to figures/<prefix>_<n>.<fmt>."""
    os.makedirs(_FIG_DIR, exist_ok=True)
    for n in plt.get_fignums():
        path = os.path.join(_FIG_DIR, f"{prefix}_{n}.{fmt}")
        plt.figure(n).savefig(path)
        log.info("Saved %s", path)
