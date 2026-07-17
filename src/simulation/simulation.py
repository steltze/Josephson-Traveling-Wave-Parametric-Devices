from __future__ import annotations

from functools import reduce
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from backends import Backend
from symbolic_solver.cell_single_mode import CellSingleMode
from numerical_solver.s_matrix import SMatrix, ABCD_to_S, cascade_all
from analysis.dispersion_relation import bloch_wavenumbers
from analysis.s_parameters import plot_s_parameters as _plot_s_params
from logger import get_logger, timer as _timer

log = get_logger(__name__)


class Simulation:
    """
    End-to-end TWPA/TWPC simulation.

    Parameters
    ----------
    cell_cls : class with a static ``build(config) -> list[CellImmitance]`` method
        Cell model factory (e.g. JTL).
    config : SimulationConfig
        All physical and numerical parameters.
    backend : Backend, backend name, or None
        Numerical backend for the ABCD-to-S conversion and Redheffer-star
        cascade (see `backends.available_backends()`). Defaults to "numpy",
        or the $TWPA_BACKEND environment variable if set.

    Examples
    --------
    >>> cfg = SimulationConfig(M=1, ks_state=[0, 1], epsilon=0.1)
    >>> sim = Simulation(JTL, cfg)
    >>> S = sim.get_s_matrix()
    >>> sim.plot_s_parameters([(3, 1), (1, 1)])
    >>> sim.plot_dispersion_relation()
    >>> plt.show()

    >>> sim = Simulation(JTL, cfg, backend="numba")  # JIT + multi-threaded cascade
    """

    def __init__(self, cell_cls, config, backend: Backend | str | None = None) -> None:
        self._cell_cls = cell_cls
        self._cfg = config
        self._backend = backend
        self._model = CellSingleMode()

        self._T_sym = None
        self._state_syms = None
        self._Zs_m_p = None
        self._Zs_m_m = None
        self._Yg_m_p = None
        self._Yg_m_m = None
        self._T_grid = None
        self._S_cells = None
        self._S_matrix: SMatrix | None = None
        self._S_matrix_normalized: bool | None = None

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
                    self._Zs_m_p,
                    self._Zs_m_m,
                    self._Yg_m_p,
                    self._Yg_m_m,
                ) = self._model.build_symbolic_transfer_matrix(
                    self._cfg.M, self._cfg.ks_state
                )
        return self._T_sym, self._state_syms

    def get_s_cells(self) -> np.ndarray:
        """
        Return the per-cell S-matrices (pre-cascade), shape (Nf, Nc, N, N).

        This is the input to `numerical_solver.s_matrix.cascade_all` inside
        `get_s_matrix`, exposed on its own so that cascade-level diagnostics
        (`analysis.checks.check_cascade_associativity`,
        `check_cascade_conditioning`, `check_backend_agreement`,
        `plot_gain_vs_ncell`, ...) can inspect the cascade before/without
        reducing it to a single total S-matrix.
        """
        if self._S_cells is None:
            T_grid = self._get_T_grid()  # (Nf, Nc, N, N)
            Nf, Nc, N, _ = T_grid.shape
            Z0 = self._cfg.Z0
            self._S_cells = ABCD_to_S(
                np.linalg.inv(T_grid.reshape(Nf * Nc, N, N)), Z0, backend=self._backend
            ).reshape(Nf, Nc, N, N)
        return self._S_cells

    def get_s_matrix(self, normalize: bool = False) -> SMatrix:
        """
        Return the cascaded S-matrix.

        Shape: (Nf, dim, dim)

        Parameters
        ----------
        normalize : bool
            If True, re-normalize to the quasi-photon-flux basis via
            ``S_ph = D⁻¹ @ S @ D``, including the kinetic-inductance
            (plasma) correction to each port's local characteristic
            impedance: weight w_k = (1 - ω_k²/ω_j²)² / ω_k, for port
            frequency ω_k and junction plasma frequency ω_j.
        """
        if self._S_matrix is None or self._S_matrix_normalized != normalize:
            with _timer("S-matrix cascade"):
                # Convert each per-cell T-matrix to S, then cascade via Redheffer star
                S_cells = self.get_s_cells()
                S_total = cascade_all(S_cells, backend=self._backend)
                self._S_matrix = SMatrix(S_total, self._cfg.Z0)

            if normalize:
                ks = np.asarray(self._cfg.ks_state)
                port_omegas_half = np.abs(self._cfg.omegas[:, None] + ks[None, :] * self._cfg.omega_pump)  # (Nf, N//2)
                port_omegas = np.concatenate([port_omegas_half, port_omegas_half], axis=1)  # (Nf, N)
                weights = 1 / np.sqrt(port_omegas)
                S_ph = self._S_matrix.array / (weights[:, None, :] / weights[:, :, None])
                self._S_matrix = SMatrix(S_ph, self._cfg.Z0)

            self._S_matrix_normalized = normalize

        return self._S_matrix

    def plot_dispersion_relation(
        self,
        cell_idx: int | None = None,
        ax: plt.Axes | None = None,
        sideband: int | None = None,
    ) -> plt.Axes:
        """
        Plot Bloch wavenumbers vs frequency for a representative unit cell.

        Parameters
        ----------
        cell_idx : int or None
            Which cell to use for the dispersion (default: middle cell).
        ax : matplotlib Axes or None
        sideband : int or None
            If given, a value from ``ks_state`` (e.g. 0 for the signal):
            only Bloch modes dominated by that sideband's voltage component
            are plotted. Default (None) plots every mode.
        """
        T_grid = self._get_T_grid()  # (Nf, Nc, dim, dim)
        idx = cell_idx if cell_idx is not None else self._cfg.ncell // 2
        T_single = T_grid[:, idx]  # (Nf, dim, dim)

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        if sideband is not None:
            alpha, k, eigenvectors = bloch_wavenumbers(T_single, return_eigenvectors=True)
            n_modes = len(self._cfg.ks_state)
            sb_idx = self._cfg.ks_state.index(sideband)
            # rows 0:n_modes of each eigenvector are the V(k) components,
            # in ks_state order — the largest one identifies the dominant sideband.
            v_weight = np.abs(eigenvectors[:, :n_modes, :])  # (Nf, n_modes, N)
            dominant = np.argmax(v_weight, axis=1)  # (Nf, N)
            mask = dominant == sb_idx
            freqs_grid = np.broadcast_to(self._cfg.freqs[:, None], k.shape)
            ax.plot(k[mask], freqs_grid[mask], ".", label=f"k={sideband}")
        else:
            alpha, k = bloch_wavenumbers(T_single)
            for mode_i in range(k.shape[1]):
                ax.plot(k[:, mode_i], self._cfg.freqs, ".", label=f"mode {mode_i}")

        ax.set_xlabel("k (rad/cell)")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(
            f"Dispersion relation  "
            f"(M={self._cfg.M}, ks={self._cfg.ks_state}, cell {idx})"
        )
        ax.legend()
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
        normalize = kwargs.pop("normalize", False)
        k = kwargs.pop("k", 0.0)

        S = self.get_s_matrix(normalize=normalize)
        return _plot_s_params(S.array, self._cfg.freqs+k*self._cfg.omega_pump/2.0/np.pi, params, ax=ax, **kwargs)

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
                    self._Zs_m_p,
                    self._Zs_m_m,
                    self._Yg_m_p,
                    self._Yg_m_m,
                    self._cfg.omegas,
                    self._cfg.omega_pump,
                    cells,
                )
        return self._T_grid
