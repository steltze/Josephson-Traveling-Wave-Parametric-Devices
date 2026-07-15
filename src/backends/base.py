from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """
    Numerical backend for the ABCD/S-matrix cascade kernels.

    An implementation supplies the two batched linear-algebra kernels the
    solver is built on: ABCD-to-S conversion and the Redheffer star
    product. `numerical_solver.s_matrix` dispatches to whichever backend
    is selected via `backends.get_backend`; every implementation must
    satisfy this contract so they stay interchangeable.

    Both methods receive arrays already coerced to complex dtype and
    batched to (Nf, N, N) / (Nf, N) — that normalization happens once in
    `numerical_solver.s_matrix`, not per backend, so an accelerator only
    has to implement the numerical kernel itself.
    """

    name: str

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
