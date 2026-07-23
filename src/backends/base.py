from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """
    Numerical backend for the transfer-matrix/ABCD/S-matrix cascade kernels.

    An implementation supplies the batched linear-algebra kernels the
    solver is built on: per-cell transfer-matrix construction, ABCD-to-S
    conversion, and the Redheffer star product. `numerical_solver.tranfer_matrix`
    and `numerical_solver.s_matrix` dispatch to whichever backend is
    selected via `backends.get_backend`; every implementation must satisfy
    this contract so they stay interchangeable.

    Methods receive arrays already coerced to complex dtype and batched to
    (Nf, N, N) / (Nf, N) — that normalization happens once in the calling
    module, not per backend, so an accelerator only has to implement the
    numerical kernel itself.
    """

    name: str

    @abstractmethod
    def single_mode_matrix(self, Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
        """
        Per-cell ABCD matrix for the "L" (series-then-shunt) topology.

        Parameters
        ----------
        Zs, Yg : ndarray, complex, shape (Nf, m, m)
            Series impedance and shunt admittance harmonic matrices.

        Returns
        -------
        ndarray, complex, shape (Nf, 2m, 2m)
        """

    @abstractmethod
    def symmetric_single_mode_matrix(self, Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
        """
        Per-cell ABCD matrix for the symmetric "pi" (shunt/2-series-shunt/2) topology.

        Parameters
        ----------
        Zs, Yg : ndarray, complex, shape (Nf, m, m)
            Series impedance and shunt admittance harmonic matrices.

        Returns
        -------
        ndarray, complex, shape (Nf, 2m, 2m)
        """

    def single_mode_matrix_grid(self, cells: list, topology: str) -> np.ndarray:
        """
        Build the per-cell, per-frequency ABCD matrix grid for a whole cascade.

        Parameters
        ----------
        cells : list of CellImmitance
            Each cell's `Zs_harm_fn` / `Yg_harm_fn` must already be arrays
            of shape (Nf, m, m) (the harmonic impedance/admittance
            matrices), not callables.
        topology : "L" or "pi"

        Returns
        -------
        ndarray, complex, shape (Nf, Ncells, 2m, 2m)

        Default implementation: one Python-level dispatch per cell (Nc
        calls total) into `single_mode_matrix` / `symmetric_single_mode_matrix`,
        each of which handles the full frequency batch at once. A backend
        whose per-cell overhead is dominated by per-call dispatch rather
        than by the work itself should override this to fuse the whole
        cell loop into a single compiled kernel instead (see
        `Backend.cascade_all`'s docstring for the analogous rationale).
        """
        Ncells = len(cells)
        Nf, m, _ = cells[0].Zs_harm_fn.shape
        T_grid = np.empty((Nf, Ncells, 2 * m, 2 * m), dtype=complex)
        fn = self.single_mode_matrix if topology == "L" else self.symmetric_single_mode_matrix
        for c, cell in enumerate(cells):
            T_grid[:, c] = fn(cell.Zs_harm_fn, cell.Yg_harm_fn)
        return T_grid

    @abstractmethod
    def abcd_to_s(self, abcd: np.ndarray, z0: np.ndarray) -> np.ndarray:
        """
        Convert a batch of ABCD matrices to S-parameters.

        Parameters
        ----------
        abcd : ndarray, complex, shape (Nf, N, N), N even
        z0   : ndarray, complex, shape (Nf, N)
            Per-frequency, per-port reference impedance.

        Returns
        -------
        ndarray, complex, shape (Nf, N, N)
        """

    @abstractmethod
    def redheffer_star(self, s2: np.ndarray, s1: np.ndarray) -> np.ndarray:
        """
        Cascade two batches of S-matrices via the Redheffer star product.

        Parameters
        ----------
        s1, s2 : ndarray, complex, shape (Nf, N, N), N even
            s1 is the left (input-side) block, s2 is the right (output-side) block.

        Returns
        -------
        ndarray, complex, shape (Nf, N, N)
        """

    def cascade_all(self, s_cells: np.ndarray) -> np.ndarray:
        """
        Cascade a stack of per-cell S-matrices into one total S-matrix.

        Parameters
        ----------
        s_cells : ndarray, complex, shape (Nf, Nc, N, N), N even
            Per-cell S-matrices for Nc cells, ordered left (input) to
            right (output).

        Returns
        -------
        ndarray, complex, shape (Nf, N, N)

        Default implementation: sequentially reduces via `redheffer_star`,
        one call per cell (Nc calls total) — this is what
        `Simulation.get_s_matrix` did directly before this method existed.
        A backend whose `redheffer_star` overhead is dominated by per-call
        dispatch rather than by the work itself (e.g. a JIT backend with
        many small cells) should override this to fuse the whole
        reduction into a single compiled kernel instead.
        """
        Nc = s_cells.shape[1]
        total = s_cells[:, 0]
        for c in range(1, Nc):
            total = self.redheffer_star(s_cells[:, c], total)
        return total

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
