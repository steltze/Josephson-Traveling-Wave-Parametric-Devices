import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from examples.utils import COLOR_PYTHON, PAPER_STYLE
from logger import get_logger, setup_logging
from models import JTLDiscrete
from simulation import Simulation, SimulationConfig

log = get_logger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")


def adiabatic_pump_calibration():
    """
    Compares S31 with and without that calibration, and plots the
    resulting epsilon(z) profile.
    """
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0, 1],
        ncell=2000,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.07,
        omega_c=3.4 * 2 * np.pi,
        v_ratio=-2.5,  # < 0 => counter-propagating
        freq_min=1,
        freq_max=8,
        n_freqs=500,
        disorder=False,
        epsilon_nramp=0,
        adiabatic_pump=True,
    )
    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi))

    sim = Simulation(JTLDiscrete, cfg, backend="numpy")
    S = sim.get_s_matrix(normalize=True).array
    S31_db = 10 * np.log10(np.abs(S[:, 2, 0]) ** 2 + 1e-15)

    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.plot(cfg.freqs, S31_db, color=COLOR_PYTHON, lw=1.5, solid_capstyle="round")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("|S|² (dB)")
        ax.set_title(f"Adiabatic pump calibration - N={cfg.ncell} cells", loc="left")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        os.makedirs(FIGURES_DIR, exist_ok=True)
        svg_path = os.path.join(FIGURES_DIR, "adiabatic_pump_calibration.svg")
        fig.savefig(svg_path)
        log.info("Saved %s", svg_path)
        png_path = os.path.join(FIGURES_DIR, "adiabatic_pump_calibration.png")
        fig.savefig(png_path, dpi=300)
        log.info("Saved %s", png_path)

    plt.show()


if __name__ == "__main__":
    setup_logging()
    adiabatic_pump_calibration()
