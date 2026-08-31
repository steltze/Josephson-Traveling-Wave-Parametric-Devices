import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from utils import COLOR_PYTHON, PAPER_STYLE

from analysis.checks import (
    check_photon_flux_conservation,
    check_transfer_matrix_determinant,
)
from dashboard import Dashboard
from logger import get_logger, setup_logging
from models import JTLDiscrete
from simulation import Simulation, SimulationConfig

log = get_logger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")


def julia_comparison(dashboard, *, M=1, ks_state=None, ncell=320, n_freqs=500, backend="numpy"):
    if ks_state is None:
        ks_state = [-1, 0, 1]
    cfg = SimulationConfig(
        Z0=50,
        M=M,
        ks_state=ks_state,
        ncell=ncell,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 530e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,  # usually smaller
        epsilon=0.0,
        phi_dc_frac=1/3,  # Phi_dc/Phi0, matches julia/josephsoncircuits_comparison.jl's Phi_dc_frac
        phi_rf_frac=0.01,
        v_ratio=-2.5,  # > 0 => co-propagating, < 0 => counter-propagating
        freq_min=1,  # GHz
        freq_max=14,  # GHz
        n_freqs=n_freqs,
        disorder=False,
        epsilon_nramp=0,  # where the peak will be
        adiabatic_pump=False,
    )

    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi))
    log.info("omega_cutoff = %.3f GHz", cfg.omega_cutoff / (2 * np.pi))

    sim = Simulation(JTLDiscrete, cfg, backend=backend, cell_topology="pi")
    S_params = sim.get_s_matrix(normalize=True).array

    T_grid = sim.get_transfer_matrix_grid()

    _, _ = check_transfer_matrix_determinant(T_grid, tolerance=1e-10)

    sim.plot_dispersion_relation()

    flux_conservation_S = check_photon_flux_conservation(
        S_params, cfg.omegas, cfg.omega_pump, cfg.ks_state
    )

    signal_left_index = ks_state.index(0)

    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))

        ax.plot(
            cfg.freqs, flux_conservation_S[:, signal_left_index],
            color=COLOR_PYTHON, lw=1.5, solid_capstyle="round",
        )
        ax.axhline(1, color="gray", ls="--", lw=0.8, zorder=0)
        ax.set_xlabel("Signal frequency (GHz)")
        ax.set_ylabel(r"$\sum_i \eta_i |S_{i,0}|^2$")
        ax.set_title("Photon-flux conservation, signal left", loc="left")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        os.makedirs(FIGURES_DIR, exist_ok=True)
        svg_path = os.path.join(FIGURES_DIR, "julia_comparison_flux_conservation.svg")
        fig.savefig(svg_path)
        log.info("Saved %s", svg_path)
        png_path = os.path.join(FIGURES_DIR, "julia_comparison_flux_conservation.png")
        fig.savefig(png_path, dpi=300)
        log.info("Saved %s", png_path)

    plt.show()

    if dashboard:
        dashboard_runs = [S_params]
        dashboard_labels = ["TWPC"]
        Dashboard(
            dashboard_runs,
            freqs=cfg.freqs,
            labels=dashboard_labels,
            ks_state=cfg.ks_state,
        ).run()
    else:
        signal_left_index = ks_state.index(0)
        signal_right_index = len(ks_state) + signal_left_index

        S21 = S_params[:, signal_right_index, signal_left_index]  # port1 -> port2, direct
        S12 = S_params[:, signal_left_index+1, signal_left_index]  # port2 -> port1, direct

        with mpl.rc_context(PAPER_STYLE):
            fig, ax = plt.subplots(figsize=(3.4, 2.6))

            ax.plot(
                cfg.freqs, 20 * np.log10(np.abs(S21)),
                color=COLOR_PYTHON, lw=1.5, solid_capstyle="round",
            )
            ax.set_xlabel("Signal frequency (GHz)")
            ax.set_ylabel("|S| (dB)")
            ax.set_title("Direct transmission (signal → signal)", loc="left")
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()

            os.makedirs(FIGURES_DIR, exist_ok=True)
            svg_path = os.path.join(FIGURES_DIR, "julia_comparison_s21.svg")
            fig.savefig(svg_path)
            log.info("Saved %s", svg_path)
            png_path = os.path.join(FIGURES_DIR, "julia_comparison_s21.png")
            fig.savefig(png_path, dpi=300)
            log.info("Saved %s", png_path)

        plt.show()



if __name__ == "__main__":
    setup_logging()
    julia_comparison(dashboard=False)
