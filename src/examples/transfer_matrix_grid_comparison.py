import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete
from numerical_solver.tranfer_matrix import single_mode_matrix_grid
from examples.utils import save_all

log = get_logger(__name__)


def _base_config(**overrides):
    cfg_kwargs = dict(
        Z0=50,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 530e-3,
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.06,
        omega_c=3.4 * 2 * np.pi,
        v_ratio=2.5,
        freq_min=1,
        freq_max=12,
        n_freqs=500,
        disorder=False,
        nramp=0,
        ncell=321,
    )
    cfg_kwargs.update(overrides)
    return SimulationConfig(**cfg_kwargs)


def _compare_one(cfg):
    """Return (T_numeric, T_symbolic, diff) for one config."""
    sim = Simulation(JTLDiscrete, cfg, cell_topology="L")
    cells = JTLDiscrete.build(cfg, cell_topology="L")

    T_sym, state_syms = sim.get_symbolic_matrix()
    dim = len(state_syms)

    T_numeric = single_mode_matrix_grid(cells, "L")
    T_symbolic = sim._model.build_cell_freq_matrices(
        T_sym,
        dim,
        cfg.M,
        cfg.ks_state,
        sim._Zs_m_p,
        sim._Zs_m_m,
        sim._Yg_m_p,
        sim._Yg_m_m,
        cfg.omegas,
        cfg.omega_pump,
        cells,
    )
    return T_numeric, T_symbolic, state_syms


def _signal_s_matrix(cfg, T_grid):
    """
    S-parameter array (Nf, N, N) for a given (pre-computed) T_grid, using the
    same cascade + photon-flux normalization as `Simulation.get_s_matrix`.
    """
    sim = Simulation(JTLDiscrete, cfg, cell_topology="L")
    sim._T_grid = T_grid  # bypass _get_T_grid(); reuse its cascade/normalize logic
    return sim.get_s_matrix().array


def _signal_left_right_ports(ks_state):
    """0-based (row, col) into the S-matrix for the ks=0 (signal) port,
    right (output) <- left (input)."""
    Nk = len(ks_state)
    idx0 = ks_state.index(0)
    return Nk + idx0, idx0


def compare_transfer_matrix_implementations():
    """
    Cross-check the two per-cell transfer-matrix builders against each other:

    - `single_mode_matrix_grid` (numerical_solver/tranfer_matrix.py): builds
      the 2n x 2n block matrix directly from each cell's pre-computed (Nf, n, n)
      Zs/Yg sideband-coupling matrices (`Component.impedance_matrix`, which
      combines the JJ inductor and its self-capacitance via an *exact*
      `np.linalg.inv` of the parallel combination). This is the path
      `Simulation._get_T_grid` actually uses.
    - `CellSingleMode.build_cell_freq_matrices` (symbolic_solver/cell_single_mode.py):
      derives the same matrix by symbolically expanding the ladder recursion
      into per-harmonic Fourier coefficients up to order M and substituting
      numeric values, explicitly discarding any Zs/Yg sideband coupling beyond
      order M.

    Both should encode the same V[n+1] = V[n] - Zs*I[n], I[n+1] = I[n] - Yg*V[n+1]
    recursion, so they're a check of whether "order M" means the same thing on
    both sides. Run at two truncation orders to see where they diverge:

    - M=1, ks_state=[0, 1] (2 sidebands): the state vector is too small to
      support any coupling beyond offset 1, so there's no room for the two
      builders to disagree.
    - M=2, ks_state=[-2,...,2] (5 sidebands): wide enough for offset-3 (e.g.
      k=-1 -> k=2) coupling to exist. `single_mode_matrix_grid` picks this up
      because the exact matrix inverse of the (nominally order-2) impedance
      operator resums the geometric series and leaks weak higher-order (~eps^3)
      terms into those entries; `build_cell_freq_matrices` does not, because it
      only ever reads bands 0..M off the exact matrix before assembling T.
    """
    cases = [
        ("M=1, ks=[0,1]", _base_config(M=1, ks_state=[0, 1], ncell=320, n_freqs=500)),
        (
            "M=2, ks=[-2..2]",
            _base_config(M=2, ks_state=[-2, -1, 0, 1, 2], ncell=320, n_freqs=500),
        ),
    ]

    plt.figure()
    for label, cfg in cases:
        T_numeric, T_symbolic, state_syms = _compare_one(cfg)

        diff = np.abs(T_numeric - T_symbolic)
        scale = np.maximum(np.abs(T_numeric), np.abs(T_symbolic))
        rel_diff = diff / np.where(scale > 0, scale, 1.0)

        log.info(
            "[%s] max abs = %.3e, max rel = %.3e, mean abs = %.3e",
            label, diff.max(), rel_diff.max(), diff.mean(),
        )

        if diff.max() > 0:
            worst = np.unravel_index(np.argmax(diff), diff.shape)
            _, _, row, col = worst
            log.info(
                "  worst entry: %s -> %s  (numeric=%s, symbolic=%s)",
                state_syms[row], state_syms[col],
                T_numeric[worst], T_symbolic[worst],
            )

        max_abs_per_freq = diff.reshape(len(cfg.freqs), -1).max(axis=1)
        plt.semilogy(cfg.freqs, np.maximum(max_abs_per_freq, 1e-18), marker="o", label=label)

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("max |T_numeric - T_symbolic| over cells/entries")
    plt.title("single_mode_matrix_grid vs. build_cell_freq_matrices")
    plt.legend()
    plt.grid(True, alpha=0.25)

    # --- signal (k=0) left-to-right transmission, both builders ---
    for label, cfg in cases:
        T_numeric, T_symbolic, _ = _compare_one(cfg)
        S_numeric = _signal_s_matrix(cfg, T_numeric)
        S_symbolic = _signal_s_matrix(cfg, T_symbolic)
        row, col = _signal_left_right_ports(cfg.ks_state)

        s_num = S_numeric[:, row, col]
        s_sym = S_symbolic[:, row, col]
        db_num = 20.0 * np.log10(np.abs(s_num))
        db_sym = 20.0 * np.log10(np.abs(s_sym))

        fig, (ax_mag, ax_diff) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
        ax_mag.plot(cfg.freqs, db_num, label="single_mode_matrix_grid (active)")
        ax_mag.plot(cfg.freqs, db_sym, "--", label="build_cell_freq_matrices (symbolic)")
        ax_mag.set_ylabel("|S| signal L->R (dB)")
        ax_mag.set_title(f"Signal (k=0) transmission, {label}")
        ax_mag.legend()
        ax_mag.grid(True, alpha=0.25)

        ax_diff.semilogy(cfg.freqs, np.maximum(np.abs(db_num - db_sym), 1e-18))
        ax_diff.set_xlabel("Frequency (GHz)")
        ax_diff.set_ylabel("|dB difference|")
        ax_diff.grid(True, alpha=0.25)

    # save_all("transfer_matrix_grid_comparison")
    plt.show()


if __name__ == "__main__":
    setup_logging()
    compare_transfer_matrix_implementations()
