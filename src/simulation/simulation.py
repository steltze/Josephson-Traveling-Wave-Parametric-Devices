from __future__ import annotations

import logging
import operator
import time
from contextlib import contextmanager
from functools import reduce
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from symbolic.cell_single_mode import CellSingleMode
from solver.s_matrix import SMatrix, ABCD_to_S, redheffer_star
from analysis.dispersion_relation import bloch_wavenumbers
from analysis.s_parameters import plot_s_parameters as _plot_s_params

log = logging.getLogger(__name__)


@contextmanager
def _timer(label: str):
    t0 = time.perf_counter()
    yield
    log.info("[%s] %.4fs", label, time.perf_counter() - t0)


class Simulation:
    """
    End-to-end TWPA/TWPC simulation.

    Parameters
    ----------
    cell_cls : class with a static ``build(config) -> list[CellImmitance]`` method
        Cell model factory (e.g. JTL).
    config : SimulationConfig
        All physical and numerical parameters.

    Examples
    --------
    >>> cfg = SimulationConfig(M=1, ks_state=[0, 1], epsilon=0.1)
    >>> sim = Simulation(JTL, cfg)
    >>> S = sim.get_s_matrix()
    >>> sim.plot_s_parameters([(3, 1), (1, 1)])
    >>> sim.plot_dispersion_relation()
    >>> plt.show()
    """

    def __init__(self, cell_cls, config) -> None:
        self._cell_cls = cell_cls
        self._cfg = config
        self._model = CellSingleMode()

        self._T_sym = None
        self._state_syms = None
        self._Zs_m = None
        self._Yg_m = None
        self._T_grid = None
        self._S_matrix: SMatrix | None = None

    def get_symbolic_matrix(self):
        """
        Build (and cache) the symbolic transfer matrix.

        Returns
        -------
        T_sym : sympy Matrix
        state_syms : list of sympy expressions
        """
        if self._T_sym is None:
            with _timer("Symbolic transfer matrix"):
                (
                    self._T_sym,
                    self._state_syms,
                    self._Zs_m,
                    self._Yg_m,
                ) = self._model.build_symbolic_transfer_matrix(
                    self._cfg.M, self._cfg.ks_state
                )
        return self._T_sym, self._state_syms

    def get_s_matrix(self) -> SMatrix:
        """
        Return the cascaded S-matrix.

        Transfer matrices are multiplied first (numerically stable for active
        systems near parametric oscillation), then a single ABCD→S conversion
        is applied.  Cell-by-cell Redheffer cascading is avoided because near
        the backward-wave phase-matching frequency the per-step denominator
        (I - S2_11 @ S1_22) becomes nearly singular.

        Shape: (Nf, dim, dim)
        """
        if self._S_matrix is None:
            with _timer("S-matrix cascade"):
                T_grid = self._get_T_grid()  # (Nf, Nc, N, N)
                Nf, Nc, N, _ = T_grid.shape
                # Convert each per-cell T-matrix to S, then cascade via Redheffer star
                S_cells = ABCD_to_S(np.linalg.inv(T_grid.reshape(Nf * Nc, N, N)), self._cfg.Z0).reshape(Nf, Nc, N, N)
                # Photon-flux normalization: S_ph[f,c,i,j] = S[f,c,i,j] * sqrt(ω_j/ω_i)
                ks = np.asarray(self._cfg.ks_state)
                # port_omegas = np.concatenate([
                #     self._cfg.omegas[:, None] + ks[None, :] * self._cfg.omega_pump,
                # ] * 2, axis=1)  # (Nf, N)
                # sqrt_omega = np.sqrt(port_omegas)  # (Nf, N)
                # S_cells = S_cells * (sqrt_omega[:, None, None, :] / sqrt_omega[:, None, :, None])
                S_total = S_cells[:, 0]
                for c in range(1, Nc):
                    S_total = redheffer_star(S_cells[:, c], S_total)
                self._S_matrix = SMatrix(S_total, self._cfg.Z0)
        return self._S_matrix

    def plot_dispersion_relation(
        self,
        cell_idx: int | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot Bloch wavenumbers vs frequency for a representative unit cell.

        Parameters
        ----------
        cell_idx : int or None
            Which cell to use for the dispersion (default: middle cell).
        ax : matplotlib Axes or None
        """
        T_grid = self._get_T_grid()  # (Nf, Nc, dim, dim)
        idx = cell_idx if cell_idx is not None else self._cfg.ncell // 2
        T_single = T_grid[:, idx]  # (Nf, dim, dim)
        alpha, k = bloch_wavenumbers(T_single)

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        f_GHz = self._cfg.freqs / 1e9
        for mode_i in range(k.shape[1]):
            ax.plot(k[:, mode_i], f_GHz, ".", label=f"mode {mode_i}")

        ax.set_xlabel("k (rad/cell)")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(
            f"Dispersion relation  "
            f"(M={self._cfg.M}, ks={self._cfg.ks_state}, cell {idx})"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        return ax

    def plot_s_parameters(
        self,
        params: Sequence[Tuple[int, int]],
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot selected S-parameters in dB vs frequency.

        Parameters
        ----------
        params : list of (i, j) 1-based index pairs, e.g. [(3,1), (1,1)]
        ax : matplotlib Axes or None
        """
        S = self.get_s_matrix()
        return _plot_s_params(S.array, self._cfg.freqs, params, ax=ax, **kwargs)

    def _get_T_grid(self) -> np.ndarray:
        """Evaluate T_sym on the full (Nf, Nc) grid; cached after first call."""
        if self._T_grid is None:
            T_sym, _ = self.get_symbolic_matrix()
            cells = self._cell_cls.build(self._cfg)
            dim = len(self._state_syms)
            with _timer("Numerical cell matrices"):
                self._T_grid = self._model.build_cell_freq_matrices(
                    T_sym,
                    dim,
                    self._cfg.M,
                    self._cfg.ks_state,
                    self._Zs_m,
                    self._Yg_m,
                    self._cfg.omegas,
                    self._cfg.omega_pump,
                    cells,
                )
        return self._T_grid
