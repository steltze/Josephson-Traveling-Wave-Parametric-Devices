import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete
from models.jtl_continuous import dispersion_linear, dispersion_linear_with_plasma
from analysis.dispersion_relation import bloch_wavenumbers
from examples.utils import save_all

log = get_logger(__name__)


def compare_dispersion_relations():
    """
    Overlay three dispersion curves on one plot:
      1. Discrete  — numerical Bloch wavenumbers from JTLDiscrete transfer matrix
                     (mirrors sim.plot_dispersion_relation: uses interior/middle cell)
      2. Continuous linear          : k = 2ω / ωc
      3. Continuous linear + plasma : k = 2ω / ωc / sqrt(1 - ω²/ωj²)
    """
    omega_cutoff = 2 * 50 / 530e-3
    omega_j      = 30 * 2 * np.pi
    # band edge of the discrete JTL: cos(ka)=-1 → solve 2(ω/ωc)²/(1-ω²/ωj²)=2
    omega_max    = omega_cutoff / np.sqrt(1.0 + (omega_cutoff / omega_j) ** 2)
    freq_max_ghz = omega_max / (2 * np.pi) * 0.999  # just below band edge

    ncell = 11
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0],           # signal only — no coupling needed for bare dispersion
        ncell=ncell,
        cell_size=10e-6,
        omega_cutoff=omega_cutoff,
        omega_pump=6.8 * 2 * np.pi,
        omega_j=omega_j,
        epsilon=0.055,            # no pump modulation
        omega_c=3.4 * 2 * np.pi,
        v_ratio=2.5,
        freq_min=0.1,
        freq_max=freq_max_ghz,
        n_freqs=1000,
        disorder=False,
        epsilon_nramp=0,
    )

    # --- discrete: same as plot_dispersion_relation — use interior (middle) cell ---
    sim = Simulation(JTLDiscrete, cfg)
    T_grid = sim._get_T_grid()              # (Nf, Nc, dim, dim)
    T_cell = T_grid[:, ncell // 2]         # middle interior cell, shape (Nf, 2, 2)
    _, k_disc = bloch_wavenumbers(T_cell)  # (Nf, 2): [+k, -k] after descending sort

    freqs  = cfg.freqs   # GHz
    omegas = cfg.omegas  # rad/GHz

    # --- analytical continuous dispersion relations ---
    k_linear = dispersion_linear(omegas, cfg.omega_cutoff, cfg.omega_j)
    k_lp     = dispersion_linear_with_plasma(omegas, cfg.omega_cutoff, cfg.omega_j)

    fig, ax = plt.subplots(figsize=(7, 5))

    # plot both ±k branches for each model
    for branch in range(2):
        ax.plot(k_disc[:, branch] / np.pi, freqs, lw=2.0, color="C0",
                label="Discrete" if branch == 0 else None)
    for sign, ls in [(+1, "--"), (-1, "--")]:
        ax.plot(sign * k_linear / np.pi, freqs, lw=1.8, ls=ls, color="C1",
                label=r"Continuous linear: $k = 2\omega/\omega_c$" if sign == 1 else None)
    # for sign, ls in [(+1, ":"), (-1, ":")]:
    #     ax.plot(sign * k_lp / np.pi, freqs, lw=1.8, ls=ls, color="C2",
    #             label=r"Linear + plasma: $k = \frac{2\omega/\omega_c}{\sqrt{1 - \omega^2/\omega_j^2}}$" if sign == 1 else None)

    ax.axvline( 1.0, color="gray", lw=0.8, ls="--", alpha=0.6, label=r"$k = \pm\pi$")
    ax.axvline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axvline( 0.0, color="gray", lw=0.5, alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel(r"$k\,/\,\pi$  (rad/cell)")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_title(
        rf"Dispersion relations  ($\omega_c/2\pi={cfg.omega_cutoff/(2*np.pi):.0f}$ GHz,"
        rf" $\omega_J/2\pi={cfg.omega_j/(2*np.pi):.0f}$ GHz)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # save_all("compare_dispersion_relations")
    plt.show()

