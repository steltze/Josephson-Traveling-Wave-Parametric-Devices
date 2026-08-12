import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete
from analysis.checks import check_transfer_matrix_determinant
from analysis.dispersion_relation import bloch_wavenumbers
from examples.utils import save_all
from symbolic_solver.cell_single_mode_symmetric import CellSingleModeSymmetric

log = get_logger(__name__)

# L needs a series-impedance-free first cell and half-shunt-capacitor ends to
# terminate the ladder symmetrically (ncell = desired sections + 1); Pi's own
# half-capacitor at each of its two ends already terminates correctly with no
# boundary special-casing (ncell = desired sections). See JTLDiscrete.build.
_NCELL = {"L": 321, "pi": 320}


def compare_cell_topologies():
    """
    Compare the plain L (series-then-shunt) unit cell against the symmetric
    Pi (shunt/2-series-shunt/2) unit cell on the same amplifier configuration:
    overlay S-parameters and dispersion relations, and check that both
    topologies keep |det(T)| == 1 per cell (losslessness).
    """
    base_cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0, 1],
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.045,
        omega_c=3.4 * 2 * np.pi,
        v_ratio=-2.5,
        freq_min=1,
        freq_max=12,
        n_freqs=500,
        disorder=False,
        epsilon_nramp=0,
    )
    signal_port, idler_port = 0, 1  # 1-based: (signal in), (idler out)

    log.info("omega_pump = %.3f GHz", base_cfg.omega_pump / (2 * np.pi))

    cfgs = {topology: replace(base_cfg, ncell=_NCELL[topology]) for topology in ("L", "pi")}
    sims = {}
    for topology, cfg in cfgs.items():
        sim = Simulation(JTLDiscrete, cfg, cell_topology=topology)
        T_grid = sim.get_transfer_matrix_grid()  # (Nf, Nc, dim, dim)
        check_transfer_matrix_determinant(T_grid, tolerance=1e-8)
        sims[topology] = sim

    # symbolic_matrix, state_syms = sim.get_symbolic_matrix()
    # a = CellSingleModeSymmetric()
    # a.export_matrix_graphic(symbolic_matrix, state_syms)

    fig, (ax_gain, ax_idler, ax_err) = plt.subplots(1, 3, figsize=(17, 5))
    s_params = {}
    for topology, color in (("L", "C0"), ("pi", "C1")):
        S = sims[topology].get_s_matrix().array  # (Nf, N, N)

        s11 = S[:, signal_port, signal_port]
        s31 = S[:, 2, signal_port]
        s_params[topology] = (s11, s31)
        ax_gain.plot(base_cfg.freqs, 20 * np.log10(np.abs(s11)), color=color, label=topology)
        ax_idler.plot(base_cfg.freqs, 20 * np.log10(np.abs(s31)), color=color, label=topology)
    ax_gain.set_title("Signal reflection |S11|")
    ax_idler.set_title("Signal->idler gain |S31|")
    for ax in (ax_gain, ax_idler):
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    s11_err = np.abs(s_params["L"][0] - s_params["pi"][0])
    s31_err = np.abs(s_params["L"][1] - s_params["pi"][1])
    ax_err.semilogy(base_cfg.freqs, s11_err, color="C2", label="|S11 error|")
    ax_err.semilogy(base_cfg.freqs, s31_err, color="C3", label="|S31 error|")
    ax_err.set_title("L vs Pi model error")
    ax_err.set_xlabel("Frequency (GHz)")
    ax_err.set_ylabel("|S(L) - S(Pi)|")
    ax_err.legend()
    ax_err.grid(True, alpha=0.3, which="both")

    fig.suptitle("L cell vs symmetric Pi cell — S-parameters")
    fig.tight_layout()

    # --- dispersion relation on a single interior cell ---
    fig2, ax = plt.subplots(figsize=(7, 5))
    for topology, color, index in (("L", "C0", 2), ("pi", "C1", 1)):
        T_grid = sims[topology].get_transfer_matrix_grid()
        T_cell = T_grid[:, index]  # (Nf, dim, dim)
        _, k = bloch_wavenumbers(T_cell)
        for branch in range(k.shape[1]):
            ax.plot(
                k[:, branch] / np.pi,
                base_cfg.freqs,
                ".",
                ms=2,
                color=color,
                label=f"{topology}" if branch == 0 else None,
            )
    ax.set_xlabel(r"$k\,/\,\pi$  (rad/cell)")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_title("L cell vs symmetric Pi cell — dispersion relation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()

    # save_all("compare_cell_topologies")
    plt.show()


if __name__ == "__main__":
    setup_logging()
    compare_cell_topologies()
