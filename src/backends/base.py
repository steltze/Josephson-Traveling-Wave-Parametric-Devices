from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """
    Numerical backend for the transfer-matrix/ABCD/S-matrix cascade kernels.

    An implementation supplies the batched linear-algebra kernels the
    solver is built on: per-cell transfer-matrix construction, ABCD-to-S
    conversion, and the Redheffer star product. Every implementation must satisfy
    this functionality so they stay interchangeable.

    Methods receive arrays already coerced to complex dtype and batched to
    (Nf, N, N) / (Nf, N).
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
    def symmetric_single_mode_matrix(
        self, Zs: np.ndarray, Yg: np.ndarray
    ) -> np.ndarray:
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

    @abstractmethod
    def slot_mode_matrix(
        self,
        Zs: np.ndarray,
        Yg: np.ndarray,
        Zs_slot: np.ndarray,
        Yg_slot: np.ndarray,
        Yi_coupling: np.ndarray,
    ) -> np.ndarray:
        """
        Per-cell ABCD matrix for the asymmetric slot-mode topology.

        State per cell: x = [V', V, I_s, I] (each an m-vector of harmonics).

        Parameters
        ----------
        Zs, Yg : ndarray, complex, shape (Nf, m, m)
            Series impedance and shunt admittance harmonic matrices, as in
            `single_mode_matrix`.
        Zs_slot, Yg_slot, Yi_coupling : ndarray, complex, shape (Nf, m, m)
            Slot-line series impedance, slot shunt admittance, and
            slot-to-signal coupling admittance harmonic matrices.

        Returns
        -------
        ndarray, complex, shape (Nf, 4m, 4m)
        """

    def single_mode_matrix_grid(self, cells: list, topology: str) -> np.ndarray:
        """
        Build the per-cell, per-frequency ABCD matrix grid for a whole cascade.

        Parameters
        ----------
        cells : list of CellImmitance
            Each cell's `Zs_harm_fn` / `Yg_harm_fn` must already be arrays
            of shape (Nf, m, m) (the harmonic impedance/admittance
            matrices), not callables. If `Zs_slot_fn` is set (slot-mode
            cells, e.g. from `JTLDiscreteSlotMode`), `Yg_slot_fn` /
            `Yi_slot_coupling_fn` must be set too, with the same shape.
        topology : "L" or "pi"
            Ignored for slot-mode cells, only "L" topology supported for now.

        Returns
        -------
        ndarray, complex, shape (Nf, Ncells, 2m, 2m), or (Nf, Ncells, 4m, 4m)
        for slot-mode cells.
        """
        Ncells = len(cells)
        Nf, m, _ = cells[0].Zs_harm_fn.shape
        if cells[0].Zs_slot_fn is not None:
            T_grid = np.empty((Nf, Ncells, 4 * m, 4 * m), dtype=complex)
            for c, cell in enumerate(cells):
                T_grid[:, c] = self.slot_mode_matrix(
                    cell.Zs_harm_fn,
                    cell.Yg_harm_fn,
                    cell.Zs_slot_fn,
                    cell.Yg_slot_fn,
                    cell.Yi_slot_coupling_fn,
                )
            return T_grid

        T_grid = np.empty((Nf, Ncells, 2 * m, 2 * m), dtype=complex)
        fn = (
            self.single_mode_matrix
            if topology == "L"
            else self.symmetric_single_mode_matrix
        )
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
        """
        Nc = s_cells.shape[1]
        total = s_cells[:, 0]
        for c in range(1, Nc):
            total = self.redheffer_star(s_cells[:, c], total)
        return total

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
