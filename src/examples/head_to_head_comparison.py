import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from logger import get_logger, setup_logging
from simulation import Simulation
from utils import COLOR_JULIA, COLOR_PYTHON, PAPER_STYLE

log = get_logger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
JULIA_SCRIPT = os.path.join(REPO_ROOT, "julia", "josephsoncircuits_comparison.jl")
JULIA_CSV = os.path.join(FIGURES_DIR, "julia_sparams.csv")


def run_python():
    """Run julia_comparison.py's real config and capture (freqs, S_params, ks_state)."""
    import julia_comparison as jc

    captured = {}
    real_get_s_matrix = Simulation.get_s_matrix

    def capturing_get_s_matrix(self, *args, **kwargs):
        result = real_get_s_matrix(self, *args, **kwargs)
        captured["S_params"] = result.array
        captured["freqs"] = self._cfg.freqs
        captured["ks_state"] = self._cfg.ks_state
        return result

    Simulation.get_s_matrix = capturing_get_s_matrix
    try:
        jc.julia_comparison(dashboard=False)
    finally:
        Simulation.get_s_matrix = real_get_s_matrix

    return captured["freqs"], captured["S_params"], captured["ks_state"]


def run_julia(skip: bool):
    if skip:
        if not os.path.isfile(JULIA_CSV):
            raise FileNotFoundError(
                f"{JULIA_CSV} doesn't exist yet -- can't --skip-julia on a first run."
            )
        log.info("Skipping Julia run, reusing existing %s", JULIA_CSV)
    else:
        log.info("Running %s (this can take a while)...", JULIA_SCRIPT)
        subprocess.run(
            ["julia", "--threads=auto", JULIA_SCRIPT], cwd=REPO_ROOT, check=True
        )

    freq_jl, S21_jl, S31_jl, S11_jl = [], [], [], []
    with open(JULIA_CSV) as f:
        for row in csv.DictReader(f):
            freq_jl.append(float(row["freq_GHz"]))
            S21_jl.append(complex(float(row["S21_re"]), float(row["S21_im"])))
            S31_jl.append(complex(float(row["S31_re"]), float(row["S31_im"])))
            S11_jl.append(complex(float(row["S11_re"]), float(row["S11_im"])))
    return (
        np.array(freq_jl),
        np.array(S21_jl),
        np.array(S31_jl),
        np.array(S11_jl),
    )


def main(skip_julia: bool):
    setup_logging()

    log.info("Running Python simulator (julia_comparison.py's config)...")
    freqs_py, S_py, ks_state = run_python()
    if ks_state != [0, 1]:
        log.warning(
            "julia_comparison.py's ks_state is %s, not [0, 1] -- the S21/S31/S11 "
            "port indices below assume [0, 1] and may be wrong.",
            ks_state,
        )
    dB = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    S21_py = S_py[:, 2, 0]  # right-signal <- left-signal: direct transmission
    S31_py = S_py[:, 1, 0]  # right-idler  <- left-signal: TWPC conversion
    S11_py = S_py[:, 0, 0]  # left-signal  <- left-signal: input return loss

    freqs_jl, S21_jl, S31_jl, S11_jl = run_julia(skip_julia)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    with mpl.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(
            2, 1, figsize=(3.4, 4.6), sharex=True,
            gridspec_kw={"height_ratios": [1.15, 1]},
        )

        for ax, (S_p, S_j, title) in zip(
            axes,
            [
                (S21_py, S21_jl, "Direct transmission (signal → signal)"),
                (S31_py, S31_jl, "Conversion (signal → idler)"),
            ],
        ):
            ax.plot(
                freqs_py, dB(S_p), label="Python (transfer-matrix)",
                color=COLOR_PYTHON, lw=1.5, zorder=2, solid_capstyle="round",
            )
            ax.plot(
                freqs_jl, dB(S_j), label="Julia (JosephsonCircuits.jl)",
                color=COLOR_JULIA, lw=1.3, zorder=3, ls=(0, (1, 1.4)),
                dash_capstyle="round",
            )
            ax.set_ylabel("|S| (dB)")
            ax.set_title(title, loc="left")
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_xlim(1, 8)
        axes[-1].set_xlabel("Signal frequency (GHz)")
        fig.align_ylabels(axes)

        # Figure-level legend above both panels -- keeps it clear of the
        # data instead of fighting for empty space inside an axes.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.93))
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.56, 1.0),
            ncol=1, frameon=False, handlelength=2.6,
        )

        svg_path = os.path.join(FIGURES_DIR, "head_to_head_comparison.svg")
        fig.savefig(svg_path)
        log.info("Saved %s", svg_path)

        png_path = os.path.join(FIGURES_DIR, "head_to_head_comparison.png")
        fig.savefig(png_path, dpi=300)
        log.info("Saved %s", png_path)

    i_py = np.argmax(dB(S31_py))
    i_jl = np.argmax(dB(S31_jl))
    log.info(
        "Python S31 peak: %.2f dB at %.3f GHz", dB(S31_py)[i_py], freqs_py[i_py]
    )
    log.info(
        "Julia  S31 peak: %.2f dB at %.3f GHz", dB(S31_jl)[i_jl], freqs_jl[i_jl]
    )


if __name__ == "__main__":
    main(skip_julia="--skip-julia" in sys.argv)
