import numpy as np

from backends import Backend, get_backend


def _resolve_backend(backend: Backend | str | None) -> Backend:
    return backend if isinstance(backend, Backend) else get_backend(backend)


def redheffer_star(
    S2: np.ndarray, S1: np.ndarray, backend: Backend | str | None = None
) -> np.ndarray:
    """
    Redheffer star product for cascading two N-port S-matrices.

    S1 is the left (input-side) block, S2 is the right (output-side) block.

    Parameters
    ----------
    S1, S2 : ndarray, shape (Nf, N, N), N even
    backend : Backend, backend name, or None
        Numerical backend to run the cascade on. Defaults to "numpy", or
        the $TWPA_BACKEND environment variable if set. See
        `backends.available_backends()`.

    Returns
    -------
    ndarray, shape (Nf, N, N)
    """
    return _resolve_backend(backend).redheffer_star(
        np.asarray(S2, dtype=complex), np.asarray(S1, dtype=complex)
    )


def cascade_all(S_cells: np.ndarray, backend: Backend | str | None = None) -> np.ndarray:
    """
    Cascade a stack of per-cell S-matrices into one total S-matrix.

    Parameters
    ----------
    S_cells : ndarray, shape (Nf, Nc, N, N), N even
        Per-cell S-matrices for Nc cells, ordered left (input) to right
        (output).
    backend : Backend, backend name, or None
        Numerical backend to run the cascade on. Defaults to "numpy", or
        the $TWPA_BACKEND environment variable if set. Backends that fuse
        the whole reduction into one compiled kernel (e.g. "numba") avoid
        Nc separate `redheffer_star` dispatches — see `Backend.cascade_all`.

    Returns
    -------
    ndarray, shape (Nf, N, N)
    """
    return _resolve_backend(backend).cascade_all(np.asarray(S_cells, dtype=complex))


def ABCD_to_S(ABCD: np.ndarray, Z0, backend: Backend | str | None = None) -> np.ndarray:
    """
    Convert ABCD transfer matrix to S-parameters.

    Z0 : scalar (ohms) for uniform real reference, OR
         array shape (N,) / (Nf, N) giving per-port reference impedance
         (complex allowed). N is the number of ports = matrix dimension.
    backend : Backend, backend name, or None
        Numerical backend to run the conversion on. Defaults to "numpy", or
        the $TWPA_BACKEND environment variable if set. See
        `backends.available_backends()`.
    """
    ABCD = np.asarray(ABCD, dtype=complex)
    batched = ABCD.ndim == 3
    if not batched:
        ABCD = ABCD[None]
    Nf, N, _ = ABCD.shape

    # --- per-port reference impedance ---
    Z0 = np.asarray(Z0, dtype=complex)
    if Z0.ndim == 0:                       # scalar -> broadcast to all ports
        Z0vec = np.full((Nf, N), Z0)
    elif Z0.ndim == 1:                     # (N,) -> same across frequency
        Z0vec = np.broadcast_to(Z0, (Nf, N))
    else:                                  # (Nf, N)
        Z0vec = Z0

    S = _resolve_backend(backend).abcd_to_s(ABCD, Z0vec)
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
    def from_ABCD(
        cls, ABCD: np.ndarray, Z0: float = 50.0, backend: Backend | str | None = None
    ) -> "SMatrix":
        """
        Construct from an ABCD transfer matrix.

        Parameters
        ----------
        ABCD : ndarray, shape (N, N) or (Nf, N, N), N even
        Z0   : reference impedance (ohms)
        backend : Backend, backend name, or None
            Numerical backend to run the conversion on (see `ABCD_to_S`).
        """
        return cls(ABCD_to_S(ABCD, Z0, backend=backend), Z0)

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

    def cascade(self, other: "SMatrix", backend: Backend | str | None = None) -> "SMatrix":
        """
        Cascade self (left/input) with other (right/output) via Redheffer star.

        Parameters
        ----------
        other : SMatrix
        backend : Backend, backend name, or None
            Numerical backend to run the cascade on (see `redheffer_star`).
        """
        if self.Z0 != other.Z0:
            raise ValueError(f"Z0 mismatch: {self.Z0} vs {other.Z0}")
        if self.N != other.N:
            raise ValueError(f"Port-count mismatch: {self.N} vs {other.N}")
        return SMatrix(redheffer_star(other._data, self._data, backend=backend), self.Z0)

    def __matmul__(self, other: "SMatrix") -> "SMatrix":
        """Cascade self (left/input) with other (right/output) via Redheffer star."""
        if not isinstance(other, SMatrix):
            return NotImplemented
        return self.cascade(other)

    def __getitem__(self, idx) -> "SMatrix":
        """Index along the frequency axis, returning a new SMatrix."""
        sliced = self._data[idx]
        if sliced.ndim == 2:
            sliced = sliced[None]
        return SMatrix(sliced, self.Z0)

    def normalize_photon_flux(
        self,
        omegas: np.ndarray,
        ks_state: list[int],
        omega_pump: float,
        port_ks: np.ndarray | None = None,
    ) -> "SMatrix":
        """
        Re-normalize to quasi-photon-flux basis for a Bloch-mode S-matrix.

        Applies  S_ph = D⁻¹ @ S @ D  so that Σᵢ |S_ph[i,j]|² = 1.

        For an ABCD→S matrix normalised to uniform Z0, the correct weight is
        the Bloch wavenumber k, not the frequency ω.  Pass port_ks to use
        k-weights; without it falls back to ω-weights (free-space convention).

        Parameters
        ----------
        omegas : ndarray, shape (Nf,)
            Signal angular frequencies.
        ks_state : list[int]
            Sideband indices.
        omega_pump : float
            Pump angular frequency.
        port_ks : ndarray, shape (Nf, N), optional
            Bloch wavenumber per port.  When given, D_kk = √k_k.
            When omitted, D_kk = √ω_k (free-space convention).
        """
        Nk = len(ks_state)
        if self.N != 2 * Nk:
            raise ValueError(
                f"Port count N={self.N} is inconsistent with 2*len(ks_state)={2 * Nk}"
            )
        if port_ks is not None:
            sqrt_w = np.sqrt(port_ks)  # (Nf, N)
        else:
            ks = np.asarray(ks_state)
            port_omegas_half = omegas[:, None] + ks[None, :] * omega_pump  # (Nf, Nk)
            port_omegas = np.concatenate([port_omegas_half, port_omegas_half], axis=1)
            sqrt_w = np.sqrt(port_omegas)  # (Nf, N)
        # (D⁻¹ S D)_ij = S_ij * √w_j / √w_i
        S_ph = self._data * (sqrt_w[:, None, :] / sqrt_w[:, :, None])
        return SMatrix(S_ph, self.Z0)

    def __repr__(self) -> str:
        return f"SMatrix(shape={self.shape}, Z0={self.Z0})"
