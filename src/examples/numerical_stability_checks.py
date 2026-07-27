import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete
from backends import available_backends
from numerical_solver.s_matrix import cascade_all
from analysis.checks import (
    check_transfer_matrix_determinant,
    check_photon_flux_conservation,
    check_pseudo_unitarity,
    check_cascade_associativity,
    check_cascade_conditioning,
    check_abcd_product_vs_s_cascade,
    check_backend_agreement,
    plot_gain_vs_ncell,
    plot_s_vs_frequency_at_cell,
)

log = get_logger(__name__)


def numerical_stability_checks(backend: str = "numpy"):
    """
    Run every check in `analysis.checks` against a single TWPA cascade.

    `ncell`/`epsilon` below are picked for non-trivial gain (large enough
    that a real instability could show up), not physical accuracy -- push
    them higher to hunt for the cell count / modulation depth where any
    of these checks starts failing.
    """
    ks_state = [0, 1]
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=ks_state,
        ncell=2300,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.04,
        omega_c=3.4 * 2 * np.pi,
        v_ratio=2.5,
        freq_min=3,
        freq_max=7  ,
        n_freqs=100,
        disorder=False,
        epsilon_nramp=0.0,
    )
    log.info(
        "ncell=%d, epsilon=%.3f, omega_pump=%.3f GHz",
        cfg.ncell,
        cfg.epsilon,
        cfg.omega_pump / (2 * np.pi),
    )

    sim = Simulation(JTLDiscrete, cfg, backend=backend)

    # --- shared cascade data ---
    T_grid = sim._get_T_grid()  # (Nf, Nc, N, N), symbolic transfer matrix per cell
    T_abcd = np.linalg.inv(T_grid)  # the ABCD convention actually fed to ABCD_to_S
    S_cells = sim.get_s_cells()  # (Nf, Nc, N, N), per-cell S, pre-cascade
    S_total = cascade_all(S_cells, backend=backend)  # raw (un-normalized) total S
    S_ph = sim.get_s_matrix(normalize=True).array  # photon-flux-normalized total S

    # 1. Per-cell transfer-matrix determinant (lossless -> |det(T)| == 1)
    check_transfer_matrix_determinant(T_abcd, tolerance=1e-10)

    # 2. Photon-flux conservation (diagonal of the Manley-Rowe identity)
    flux = check_photon_flux_conservation(S_ph, cfg.omegas, cfg.omega_pump, cfg.ks_state)
    log.test("Photon-flux conservation: max |flux - eta| = %.3e", np.abs(np.abs(flux) - 1).max())

    # 3. Full pseudo-unitarity matrix (diagonal + off-diagonal terms)
    check_pseudo_unitarity(S_ph, cfg.omegas, cfg.omega_pump, cfg.ks_state)

    # 4. Split-cascade associativity
    check_cascade_associativity(S_cells, backend=backend)

    # 5. Redheffer-star conditioning across the cascade
    cond = check_cascade_conditioning(S_cells, backend=backend)

    # 6. Raw-ABCD-product cross-check against the Redheffer-star cascade
    check_abcd_product_vs_s_cascade(T_abcd, S_total, Z0=cfg.Z0, backend=backend)

    # 7. Backend agreement (only backends that actually import)
    check_backend_agreement(S_cells, backends=tuple(available_backends()))

    # --- diagnostic plots ---
    fig, (ax_cond, ax_gain, ax_s) = plt.subplots(1, 3, figsize=(17, 4.5))

    ax_cond.semilogy(np.arange(1, cond.shape[1] + 1), cond[45], marker=".")
    ax_cond.set_xlabel("Cascade step (cell index)")
    ax_cond.set_ylabel(r"cond$(I - S_{2,11} S_{1,22})$")
    ax_cond.set_title(f"Redheffer-star conditioning (Gap center: {cfg.freqs[45]:.2f} GHz)")
    ax_cond.grid(True, alpha=0.3, which="both")

    plot_gain_vs_ncell(S_cells, backend=backend, ax=ax_gain)
    ax_gain.set_title(f"Gain vs cell count (Gap center: {cfg.freqs[45]:.2f} GHz)")

    Nc = S_cells.shape[1]
    plot_s_vs_frequency_at_cell(
        S_cells,
        cfg.freqs,
        port_out=1,
        port_in=0,
        cell_indices=[Nc // 10, 2 * Nc // 6, Nc-1],
        backend=backend,
        ax=ax_s,
    )
    ax_s.set_title("S vs frequency, partial cascades")

    fig.tight_layout()


if __name__ == "__main__":
    setup_logging()
    numerical_stability_checks()
    plt.show()
