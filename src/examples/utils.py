import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from logger import get_logger

log = get_logger(__name__)

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Validated categorical palette (color-blind-safe, checked with dataviz's
# validate_palette.js): slot 1 (blue) / slot 2 (orange), fixed identity order.
COLOR_PYTHON = "#2a78d6"
COLOR_JULIA = "#eb6834"

# Paper-figure style: apply with `with mpl.rc_context(PAPER_STYLE):` around a
# specific figure rather than globally, so it doesn't affect other plots.
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.titlesize": 10,
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.5,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def save_all(prefix: str, fmt: str = "pdf") -> None:
    """Save every open figure to figures/<prefix>_<n>.<fmt>."""
    os.makedirs(_FIG_DIR, exist_ok=True)
    for n in plt.get_fignums():
        path = os.path.join(_FIG_DIR, f"{prefix}_{n}.{fmt}")
        plt.figure(n).savefig(path)
        log.info("Saved %s", path)
