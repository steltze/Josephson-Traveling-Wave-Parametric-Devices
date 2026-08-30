from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.dispersion_relation import bloch_wavenumbers
from backends import Backend
from logger import get_logger
from logger import timer as _timer
from models.electrical_elements import multipump_frequency_grid
from numerical_solver.s_matrix import ABCD_to_S, SMatrix, cascade_all, terminate_ports
from numerical_solver.tranfer_matrix import single_mode_matrix_grid

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
    cell_topology : "L" or "pi"
        Unit-cell transfer-matrix topology. "L" (default) is the plain
        series-then-shunt cell. "pi" is the symmetric Pi
        (shunt/2-series-shunt/2) cell.

    Examples
    --------
    >>> cfg = SimulationConfig(M=1, ks_state=[0, 1], epsilon=0.1)
    >>> sim = Simulation(JTL, cfg)
    >>> S = sim.get_s_matrix()
    >>> sim.plot_s_parameters([(3, 1), (1, 1)])
    >>> sim.plot_dispersion_relation()
    >>> plt.show()

    >>> sim = Simulation(JTL, cfg, backend="numba")  # JIT + multi-threaded cascade
    >>> sim = Simulation(JTL, cfg, cell_topology="pi")  # symmetric Pi cell
    """

    _CELL_TOPOLOGIES = ("L", "pi")

    def __init__(
        self,
        cell_cls,
        config,
        backend: Backend | str | None = None,
        cell_topology: str = "pi",
    ) -> None:
        self._cell_cls = cell_cls
        self._cfg = config
        self._backend = backend
        if cell_topology not in self._CELL_TOPOLOGIES:
            raise ValueError(
                f"Unknown cell_topology {cell_topology!r}; "
                f"expected one of {sorted(self._CELL_TOPOLOGIES)}"
            )
        self._cell_topology = cell_topology

        self._T_grid = None
        self._S_cells = None
        self._S_matrix: SMatrix | None = None
        self._S_matrix_normalized: bool | None = None
        self._S_matrix_raw_full: np.ndarray | None = None

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
            T_grid = self.get_transfer_matrix_grid()  # (Nf, Nc, N, N)
            Nf, Nc, N, _ = T_grid.shape
            Z0 = self._cfg.Z0
            self._S_cells = ABCD_to_S(
                np.linalg.inv(T_grid.reshape(Nf * Nc, N, N)), Z0, backend=self._backend
            ).reshape(Nf, Nc, N, N)
        return self._S_cells

    def _port_omegas_half(self) -> np.ndarray:
        """Tracked sideband frequencies at one physical end, shape (Nf, n_half)."""
        if isinstance(self._cfg.omega_pump, list):
            # Multi-pump: tracked state is the (k_1,...,k_P) tensor lattice,
            # not a flat ks_state -- build the same per-state frequency grid
            # JTLDiscreteMultiPump uses, via multipump_frequency_grid
            # (kron/itertools.product order), one signal frequency at a time.
            return np.abs(
                np.stack(
                    [
                        multipump_frequency_grid(
                            ws, self._cfg.omega_pump, self._cfg.Kmax
                        )[0]
                        for ws in self._cfg.omegas
                    ]
                )
            )
        ks = np.asarray(self._cfg.ks_state)
        return np.abs(self._cfg.omegas[:, None] + ks[None, :] * self._cfg.omega_pump)

    def _photon_flux_weights(self, N: int) -> np.ndarray:
        """
        Photon-flux normalization weights.
        """
        port_omegas_half = self._port_omegas_half()
        n_half = port_omegas_half.shape[-1]
        blocks_per_side, remainder = divmod(N, 2 * n_half)
        if remainder != 0:
            raise ValueError(
                f"S-matrix has {N} ports, not a multiple of "
                f"2*{n_half} tracked sideband frequencies -- can't "
                f"build the photon-flux weight vector."
            )
        port_omegas_one_side = np.tile(
            port_omegas_half, (1, blocks_per_side)
        )  # (Nf, N//2)
        port_omegas = np.concatenate(
            [port_omegas_one_side, port_omegas_one_side], axis=1
        )  # (Nf, N)
        return 1 / np.sqrt(port_omegas)

    def get_s_matrix(self, normalize: bool = True) -> SMatrix:
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
                weights = self._photon_flux_weights(self._S_matrix.array.shape[-1])
                S_ph = self._S_matrix.array / (
                    weights[:, None, :] / weights[:, :, None]
                )
                self._S_matrix = SMatrix(S_ph, self._cfg.Z0)

            self._S_matrix_normalized = normalize

        return self._S_matrix

    def get_s_matrix_slot_terminated(
        self, gamma: complex, normalize: bool = True
    ) -> SMatrix:
        """
        Cascaded S-matrix with the slot line's own two ports (input- and
        output-side) eliminated via a reflective one-port termination,
        instead of exposed as measurable 50 Ohm ports.

        Physically appropriate whenever the slot/parasitic mode has no real
        external connection (e.g. the CPW slotline mode on a chip whose
        ground planes are one continuous conductor): it reflects off the
        chip edges rather than radiating into a matched load, so its effect
        on the main line shows up as resonant features in frequency, not
        broadband loss. Treating those ports as matched-50-Ohm (as plain
        `get_s_matrix` does) instead invents an absorbing channel that
        isn't physically there, making the main-line S-parameters look
        lossy for a reason that isn't real loss.

        Only valid for cascades whose cells carry slot fields (e.g.
        `JTLDiscreteSlotMode`): N must be 4x the tracked sideband count.
        The raw (un-terminated, un-normalized) cascade is cached, so
        calling this repeatedly with different `gamma` to compare short vs.
        open vs. matched is cheap -- no cascade recompute.

        Parameters
        ----------
        gamma : complex
            Reflection coefficient at the slot line's two physical ends:
              -1 : short (grounds strapped together / continuous, V'=0)
              +1 : open (grounds not connected there, I_s=0)
               0 : matched 50 Ohm (no reflection -- for comparison against
                   short/open; equivalent in effect to `get_s_matrix`'s
                   plain 4m-port result, but reduced to the same 2m-port
                   shape as short/open so all three are directly comparable)
        normalize : bool
            Same photon-flux normalization as `get_s_matrix`, applied to
            the reduced (main-line-only) S-matrix after termination.

        Returns
        -------
        SMatrix, shape (Nf, 2m, 2m), ports ordered [main_in, main_out]
        (same layout as an ordinary non-slot-mode `get_s_matrix` result).
        """
        if not self._cfg.include_slot_modes:
            log.error("Slot mode analysis not activated...exiting.")
            exit()

        if self._S_matrix_raw_full is None:
            with _timer("S-matrix cascade"):
                S_cells = self.get_s_cells()
                self._S_matrix_raw_full = cascade_all(S_cells, backend=self._backend)

        S_raw = self._S_matrix_raw_full  # (Nf, N, N), un-normalized, Z0-referenced
        N = S_raw.shape[-1]
        m = len(self._cfg.ks_state)
        if isinstance(self._cfg.omega_pump, list) or N != 4 * m:
            raise ValueError(
                "get_s_matrix_slot_terminated needs a slot-mode, single-pump "
                f"cascade with N = 4*len(ks_state) ports; got N={N}."
            )
        # Port layout (see JTLDiscreteSlotMode / slot_mode_matrix):
        # [slot_in(m), main_in(m), slot_out(m), main_out(m)].
        slot_idx = np.r_[0:m, 2 * m : 3 * m]

        S_reduced = terminate_ports(
            S_raw, slot_idx, gamma
        )  # (Nf, 2m, 2m): [main_in, main_out]

        if normalize:
            weights = self._photon_flux_weights(S_reduced.shape[-1])
            S_reduced = S_reduced / (weights[:, None, :] / weights[:, :, None])

        return SMatrix(S_reduced, self._cfg.Z0)

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
        T_grid = self.get_transfer_matrix_grid()  # (Nf, Nc, dim, dim)
        idx = cell_idx if cell_idx is not None else self._cfg.ncell // 2
        T_single = T_grid[:, idx]  # (Nf, dim, dim)

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        if sideband is not None:
            alpha, k, eigenvectors = bloch_wavenumbers(
                T_single, return_eigenvectors=True
            )
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

    def get_transfer_matrix_grid(self) -> np.ndarray:
        """Evaluate T_sym on the full (Nf, Nc) grid; cached after first call."""
        if self._T_grid is None:
            cells = self._cell_cls.build(self._cfg, cell_topology=self._cell_topology)
            with _timer("Numerical cell matrices"):
                self._T_grid = single_mode_matrix_grid(
                    cells,
                    self._cell_topology,
                    backend=self._backend,
                )
        return self._T_grid
