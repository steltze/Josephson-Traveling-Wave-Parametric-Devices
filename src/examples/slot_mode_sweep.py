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


def slot_mode_sweep1():
    """
    Grid-sweep the slot line's own cutoff (omega_cutoff_slot_mode, as a
    fraction of the main JTL's omega_cutoff) against the main<->slot
    coupling strength (Ci_coupling, as a fraction of the slot line's own
    shunt capacitance C_slot_mode -- see JTLDiscreteSlotMode.build), and
    plot S{j}{index} for each of the resulting 15 configs.
    """
    ks_state = [0, 1]
    M = 1
    ncell = 320
    base_cfg = SimulationConfig(
        Z0=50,
        M=M,
        ks_state=ks_state,
        ncell=ncell,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.04,
        omega_c=3.4 * 2 * np.pi,  # gap: well in the S parameter from signal to transmission
        v_ratio=-2.5,  # > 0 => co-propagating, < 0 => counter-propagating
        freq_min=1,  # GHz
        freq_max=12,  # GHz
        n_freqs=500,
        disorder=False,
        epsilon_nramp=0,
        adiabatic_pump=False,
    )
    log.info("omega_pump = %.3f GHz", base_cfg.omega_pump / (2 * np.pi))
    log.info("omega_cutoff = %.3f GHz", base_cfg.omega_cutoff / (2 * np.pi))

    # omega_cutoff_slot_mode = slot_frac * omega_cutoff, 5 points from 0.2x to 5x.
    slot_fracs = np.geomspace(0.2, 5.0, 5)

    # C_slot_mode = 2/(omega_cutoff_slot_mode * Z0), Ci_coupling = 2/(omega_cutoff_coupling * Z0)
    # (same L-C-ladder formula, see JTLDiscreteSlotMode.build) -- so
    # Ci_coupling = ci_ratio * C_slot_mode reduces to a plain cutoff ratio:
    #   omega_cutoff_coupling = omega_cutoff_slot_mode / ci_ratio
    ci_ratios = np.array([0.1, 1.0, 5.0])  # Ci_coupling as a fraction of C_slot_mode

    fig, axes = plt.subplots(
        len(ci_ratios), len(slot_fracs),
        figsize=(4 * len(slot_fracs), 3 * len(ci_ratios)),
        sharex=True, sharey=True,
    )

    for row, ci_ratio in enumerate(ci_ratios):
        for col, slot_frac in enumerate(slot_fracs):
            omega_cutoff_slot_mode = slot_frac * base_cfg.omega_cutoff
            omega_cutoff_coupling = base_cfg.omega_cutoff / ci_ratio

            cfg = replace(
                base_cfg,
                omega_cutoff_slot_mode=omega_cutoff_slot_mode,
                omega_cutoff_coupling=omega_cutoff_coupling,
            )
            sim = Simulation(JTLDiscreteSlotMode, cfg, backend="numpy")
            S_ph = sim.get_s_matrix(normalize=True).array

            T_grid = sim._get_T_grid()
            check_transfer_matrix_determinant(T_grid, tolerance=1e-8)

            ax = axes[row, col]
            for j in [3, 6]:
                index = 2
                ax.plot(
                    cfg.freqs,
                    (10 * np.log10(np.abs(S_ph) ** 2))[:, j, index],
                    label=f"S{j}{index}",
                )
            ax.plot(
                cfg.freqs,
                (10 * np.log10(np.abs(S_ph) ** 2))[:, index, 6],
                label=f"S{index}6",
            )
            ax.set_title(f"slot={slot_frac:.2f}x, Ci={ci_ratio:.2f}x", fontsize=10)
            ax.grid(True, alpha=0.3)
            if row == len(ci_ratios) - 1:
                ax.set_xlabel("Frequency (GHz)")
            if col == 0:
                ax.set_ylabel("|S|^2 (dB)")

    axes[0, 0].legend()
    fig.suptitle(
        f"Slot-mode sweep: omega_cutoff_slot_mode/omega_cutoff (columns) "
        f"vs Ci_coupling/C_slot_mode (rows), ncells={ncell}"
    )
    fig.tight_layout()

    save_all("slot_mode_sweep")
    plt.show()

def slot_mode_sweep2():
    import numpy as np
    from simulation import SimulationConfig, Simulation
    from models import JTLDiscreteSlotMode
    from numerical_solver.s_matrix import terminate_ports

    cfg = SimulationConfig(
        Z0=50, M=1, ks_state=[0, 1], ncell=320, cell_size=10e-6,
        omega_cutoff=2*50/540e-3, omega_pump=6.8*2*np.pi, omega_j=60*2*np.pi,
        epsilon=0.04, omega_c=3.4*2*np.pi, v_ratio=-2.5,
        freq_min=1, freq_max=12, n_freqs=500, disorder=False, epsilon_nramp=0,
        adiabatic_pump=False,
        omega_cutoff_slot_mode=1.3*(2*50/540e-3),
        omega_cutoff_coupling=280*2*np.pi,
    )
    sim = Simulation(JTLDiscreteSlotMode, cfg, backend='numpy')


    slot_fracs = np.geomspace(0.2, 5.0, 5)

    # C_slot_mode = 2/(omega_cutoff_slot_mode * Z0), Ci_coupling = 2/(omega_cutoff_coupling * Z0)
    # (same L-C-ladder formula, see JTLDiscreteSlotMode.build) -- so
    # Ci_coupling = ci_ratio * C_slot_mode reduces to a plain cutoff ratio:
    #   omega_cutoff_coupling = omega_cutoff_slot_mode / ci_ratio
    ci_ratios = np.array([0.1, 1.0, 5.0])  # Ci_coupling as a fraction of C_slot_mode

    fig, axes = plt.subplots(
        len(ci_ratios), len(slot_fracs),
        figsize=(4 * len(slot_fracs), 3 * len(ci_ratios)),
        sharex=True, sharey=True,
    )
    results = {}
    for gamma, name in [(1, 'open')]:

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
                    # S_ph = sim.get_s_matrix(normalize=True).array
                    S_ph = sim.get_s_matrix_slot_terminated(gamma=gamma, normalize=True).array

                    T_grid = sim._get_T_grid()
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

    # unitarity checks
    # for name, expect_unitary in [('short', True), ('open', True), ('matched', False)]:
    #     col_sums = np.sum(np.abs(results[name])**2, axis=1)
    #     max_dev = np.max(np.abs(col_sums - 1.0))
    #     print(f'{name}: max |colsum-1| = {max_dev:.3e}  (expect ~0: {expect_unitary})')

    
    # # cross-check terminate_ports' gamma==0 fast path against the general formula
    # N = sim._S_matrix_raw_full.shape[-1]
    # m = len(cfg.ks_state)
    # slot_idx = np.r_[0:m, 2*m:3*m]
    # kept_idx = np.setdiff1d(np.arange(N), slot_idx)
    # S_raw = sim._S_matrix_raw_full
    # S_AA = S_raw[:, kept_idx[:,None], kept_idx[None,:]]
    # S_AB = S_raw[:, kept_idx[:,None], slot_idx[None,:]]
    # S_BA = S_raw[:, slot_idx[:,None], kept_idx[None,:]]
    # S_BB = S_raw[:, slot_idx[:,None], slot_idx[None,:]]
    # k = len(slot_idx)
    # eye = np.broadcast_to(np.eye(k, dtype=complex), (S_raw.shape[0], k, k))
    # gamma_eps = 1e-12
    # X = np.linalg.solve(eye - gamma_eps*S_BB, S_BA)
    # S_general_at_zero = S_AA + gamma_eps*(S_AB @ X)
    # S_fastpath = terminate_ports(S_raw, slot_idx, 0)
    # print('gamma=0 fast-path vs general-formula-at-eps max diff:', np.max(np.abs(S_general_at_zero - S_fastpath)))

    # # complex gamma sanity (not just +-1/0) -- e.g. a lossy/partial termination
    # S_complex = terminate_ports(S_raw, slot_idx, 0.5+0.2j)
    # plt.plot()
    # print('complex gamma: finite =', np.all(np.isfinite(S_complex)), ' shape =', S_complex.shape)


if __name__ == "__main__":
    setup_logging()
    slot_mode_sweep2()
