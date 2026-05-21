import numpy as np


def redheffer_star(S2: np.ndarray, S1: np.ndarray) -> np.ndarray:
    """
    Redheffer star product for cascading two N-port S-matrices.

    S1 is the left (input-side) block, S2 is the right (output-side) block.

    Parameters
    ----------
    S1, S2 : ndarray, shape (Nf, N, N), N even

    Returns
    -------
    ndarray, shape (Nf, N, N)
    """
    Nf, N, _ = S1.shape
    k = N // 2

    S1_11 = S1[:, :k, :k]
    S1_12 = S1[:, :k, k:]
    S1_21 = S1[:, k:, :k]
    S1_22 = S1[:, k:, k:]

    S2_11 = S2[:, :k, :k]
    S2_12 = S2[:, :k, k:]
    S2_21 = S2[:, k:, :k]
    S2_22 = S2[:, k:, k:]

    # Solve (I - S2_11 @ S1_22) @ X = [S2_11 @ S1_21, S2_12]
    A1 = np.eye(k)[None] - S2_11 @ S1_22
    X1 = np.linalg.solve(A1, np.concatenate([S2_11 @ S1_21, S2_12], axis=-1))

    # Solve (I - S1_22 @ S2_11) @ X = [S1_21, S1_22 @ S2_12]
    A2 = np.eye(k)[None] - S1_22 @ S2_11
    X2 = np.linalg.solve(A2, np.concatenate([S1_21, S1_22 @ S2_12], axis=-1))

    S = np.empty((Nf, N, N), dtype=complex)
    S[:, :k, :k] = S1_11 + S1_12 @ X1[:, :, :k]
    S[:, :k, k:] = S1_12 @ X1[:, :, k:]
    S[:, k:, :k] = S2_21 @ X2[:, :, :k]
    S[:, k:, k:] = S2_22 + S2_21 @ X2[:, :, k:]

    return S


def ABCD_to_S(ABCD: np.ndarray, Z0: float) -> np.ndarray:
    """
    Convert ABCD transfer matrix to S-parameters.

    Parameters
    ----------
    ABCD : ndarray, shape (N, N) or (Nf, N, N), N even
    Z0   : reference impedance (ohms)

    Returns
    -------
    ndarray, same shape as ABCD
    """
    batched = ABCD.ndim == 3
    if not batched:
        ABCD = ABCD[None]

    Nf, N, _ = ABCD.shape
    k = N // 2

    A = ABCD[:, :k, :k]
    B = ABCD[:, :k, k:]
    C = ABCD[:, k:, :k]
    D = ABCD[:, k:, k:]

    # Solve C @ [Cinv, CinvD] = [I, D] — avoids explicit inverse
    eye = np.broadcast_to(np.eye(k, dtype=complex), (Nf, k, k))
    Cinv_CinvD = np.linalg.solve(C, np.concatenate([eye, D], axis=-1))
    Cinv = Cinv_CinvD[:, :, :k]
    CinvD = Cinv_CinvD[:, :, k:]

    Z = np.empty((Nf, N, N), dtype=complex)
    Z[:, :k, :k] = A @ Cinv
    Z[:, :k, k:] = A @ CinvD - B
    Z[:, k:, :k] = Cinv
    Z[:, k:, k:] = CinvD

    # S = (Z - Z0·I) @ (Z + Z0·I)^{-1}
    # Gref = (1/√Z0)·I cancels identically → dropped
    # Right-solve: S·Zp = Zm  →  Zp^T·S^T = Zm^T
    Z0I = Z0 * np.eye(N, dtype=complex)
    S = np.linalg.solve(
        (Z + Z0I).swapaxes(-1, -2),
        (Z - Z0I).swapaxes(-1, -2),
    ).swapaxes(-1, -2)

    return S[0] if not batched else S


class SMatrix:
    """
    Frequency-batched S-parameter matrix.

    Internal storage is always (Nf, N, N) complex, even for a single frequency.
    N must be even; the matrix is partitioned into four (N/2 x N/2) blocks.
    """

    def __init__(self, data: np.ndarray, Z0: float = 50.0) -> None:
        data = np.asarray(data, dtype=complex)
        if data.ndim == 2:
            data = data[None]
        if data.ndim != 3 or data.shape[1] != data.shape[2] or data.shape[1] % 2 != 0:
            raise ValueError(
                f"data must be (N, N) or (Nf, N, N) with N even, got {data.shape}"
            )
        self._data = data
        self.Z0 = float(Z0)

    @classmethod
    def from_ABCD(cls, ABCD: np.ndarray, Z0: float = 50.0) -> "SMatrix":
        """
        Construct from an ABCD transfer matrix.

        Parameters
        ----------
        ABCD : ndarray, shape (N, N) or (Nf, N, N), N even
        Z0   : reference impedance (ohms)
        """
        return cls(ABCD_to_S(ABCD, Z0), Z0)

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
        """Port count."""
        return self._data.shape[1]

    @property
    def _k(self) -> int:
        return self.N // 2

    @property
    def S11(self) -> np.ndarray:
        return self._data[:, : self._k, : self._k]

    @property
    def S12(self) -> np.ndarray:
        return self._data[:, : self._k, self._k :]

    @property
    def S21(self) -> np.ndarray:
        return self._data[:, self._k :, : self._k]

    @property
    def S22(self) -> np.ndarray:
        return self._data[:, self._k :, self._k :]

    def __matmul__(self, other: "SMatrix") -> "SMatrix":
        """Cascade self (left/input) with other (right/output) via Redheffer star."""
        if not isinstance(other, SMatrix):
            return NotImplemented
        if self.Z0 != other.Z0:
            raise ValueError(f"Z0 mismatch: {self.Z0} vs {other.Z0}")
        if self.N != other.N:
            raise ValueError(f"Port-count mismatch: {self.N} vs {other.N}")
        return SMatrix(redheffer_star(other._data, self._data), self.Z0)

    def __getitem__(self, idx) -> "SMatrix":
        """Index along the frequency axis, returning a new SMatrix."""
        sliced = self._data[idx]
        if sliced.ndim == 2:
            sliced = sliced[None]
        return SMatrix(sliced, self.Z0)

    def __repr__(self) -> str:
        return f"SMatrix(shape={self.shape}, Z0={self.Z0})"
