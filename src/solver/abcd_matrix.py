from __future__ import annotations

import numpy as np


class ABCDMatrix:
    """
    Frequency-batched ABCD (transfer) matrix.

    Internal storage is always (Nf, N, N) complex, even for a single frequency.
    N must be even; the matrix is partitioned into four (N/2 x N/2) blocks:

        [Vn_1 ]   [A B] [Vn ]
        [In_1 ] = [C D] [In]

    Cascading: T_total = T1 @ T2.
    """

    def __init__(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=complex)
        if data.ndim == 2:
            data = data[None]
        if data.ndim != 3 or data.shape[1] != data.shape[2] or data.shape[1] % 2 != 0:
            raise ValueError(
                f"data must be (N, N) or (Nf, N, N) with N even, got {data.shape}"
            )
        self._data = data

    @property
    def array(self) -> np.ndarray:
        """Underlying (Nf, N, N) array."""
        return self._data

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._data.shape

    @property
    def Nf(self) -> int:
        """Number of frequency points."""
        return self._data.shape[0]

    @property
    def N(self) -> int:
        """Total port count (2 × number of mode-ports per side)."""
        return self._data.shape[1]

    @property
    def half_matrix(self) -> int:
        return self.N // 2

    @property
    def A(self) -> np.ndarray:
        return self._data[:, : self.half_matrix, : self.half_matrix]

    @property
    def B(self) -> np.ndarray:
        return self._data[:, : self.half_matrix, self.half_matrix :]

    @property
    def C(self) -> np.ndarray:
        return self._data[:, self.half_matrix :, : self.half_matrix]

    @property
    def D(self) -> np.ndarray:
        return self._data[:, self.half_matrix :, self.half_matrix :]

    def __matmul__(self, other: "ABCDMatrix") -> "ABCDMatrix":
        """Cascade self (left/input) with other (right/output): T = self @ other."""
        if not isinstance(other, ABCDMatrix):
            return NotImplemented
        if self.N != other.N:
            raise ValueError(f"Port-count mismatch: {self.N} vs {other.N}")
        return ABCDMatrix(self._data @ other._data)

    def __getitem__(self, idx) -> "ABCDMatrix":
        """Index along the frequency axis, returning a new ABCDMatrix."""
        sliced = self._data[idx]
        if sliced.ndim == 2:
            sliced = sliced[None]
        return ABCDMatrix(sliced)

    @classmethod
    def from_cell_grid(cls, data: np.ndarray) -> "ABCDMatrix":
        """
        Cascade a (Nf, Nc, N, N) cell grid into a single (Nf, N, N) ABCD matrix.

        For each frequency slice the result is:
            T[f] = data[f, 0] @ data[f, 1] @ ... @ data[f, Nc-1]
        where cell 0 is the input-side cell.

        Parameters
        ----------
        data : ndarray, shape (Nf, Nc, N, N)

        Returns
        -------
        ABCDMatrix of shape (Nf, N, N)
        """
        data = np.asarray(data, dtype=complex)
        if data.ndim != 4 or data.shape[2] != data.shape[3]:
            raise ValueError(
                f"data must be (Nf, Nc, N, N), got {data.shape}"
            )
        result = data[:, 0]
        for c in range(1, data.shape[1]):
            result = result @ data[:, c]
        return cls(result)

    def to_S(self, Z0: float = 50.0) -> "SMatrix":
        """
        Convert to an SMatrix.

        Parameters
        ----------
        Z0 : reference impedance in ohms (default 50)

        Returns
        -------
        SMatrix of shape (Nf, N, N)
        """
        from src.solver.s_matrix import SMatrix
        return SMatrix.from_ABCD(self._data, Z0)

    def __repr__(self) -> str:
        return f"ABCDMatrix(shape={self.shape})"
