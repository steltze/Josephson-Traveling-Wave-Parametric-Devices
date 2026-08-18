import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscreteSlotMode
from analysis.checks import check_transfer_matrix_determinant
from examples.utils import save_all

log = get_logger(__name__)


def slot_mode_sweep():
    import numpy as np
    from simulation import SimulationConfig, Simulation
    from models import JTLDiscreteSlotMode
    from numerical_solver.s_matrix import terminate_ports

    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0, 1],
        ncell=320,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.04,
        omega_c=3.4 * 2 * np.pi,
        v_ratio=-2.5,
        freq_min=1,
        freq_max=12,
        n_freqs=500,
        disorder=False,
        epsilon_nramp=0,
        adiabatic_pump=False,
        include_slot_modes=True,
        omega_cutoff_slot_mode=1.3 * (2 * 50 / 540e-3),
        omega_cutoff_coupling=280 * 2 * np.pi,
    )

    slot_fracs = np.array([1.0, 5.0])
    ci_ratios = np.array([0.1, 0.5, 1.0])  # Ci_coupling as a fraction of C_slot_mode

    fig, axes = plt.subplots(
        len(ci_ratios),
        len(slot_fracs),
        figsize=(4 * len(slot_fracs), 3 * len(ci_ratios)),
        sharex=True,
        sharey=True,
    )
    for gamma, name in [(1, "open")]:
        for row, ci_ratio in enumerate(ci_ratios):
            for col, slot_frac in enumerate(slot_fracs):
                omega_cutoff_slot_mode = slot_frac * cfg.omega_cutoff
                omega_cutoff_coupling = cfg.omega_cutoff / ci_ratio

                cfg = replace(
                    cfg,
                    omega_cutoff_slot_mode=omega_cutoff_slot_mode,
                    omega_cutoff_coupling=omega_cutoff_coupling,
                )
                sim = Simulation(JTLDiscreteSlotMode, cfg, backend="numpy")
                S_ph = sim.get_s_matrix_slot_terminated(
                    gamma=gamma, normalize=True
                ).array

                T_grid = sim.get_transfer_matrix_grid()
                check_transfer_matrix_determinant(T_grid, tolerance=1e-8)

                ax = axes[row, col]
                for j in [1, 2]:
                    index = 0
                    ax.plot(
                        cfg.freqs,
                        (10 * np.log10(np.abs(S_ph[:, j, index]) ** 2)),
                        label=f"S{j}{index}",
                    )

                ax.set_title(f"slot={slot_frac:.2f}x, Ci={ci_ratio:.2f}x", fontsize=10)
                ax.grid(True, alpha=0.3)
                if row == len(ci_ratios) - 1:
                    ax.set_xlabel("Frequency (GHz)")
                if col == 0:
                    ax.set_ylabel("|S|^2 (dB)")

        axes[0, 0].legend()
        fig.tight_layout()

    save_all("slot_mode_sweep")
    plt.show()


if __name__ == "__main__":
    setup_logging()
    slot_mode_sweep()
