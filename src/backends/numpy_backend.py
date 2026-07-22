from __future__ import annotations

import numpy as np

from backends.base import Backend


class NumpyBackend(Backend):
    """Reference backend using NumPy's batched linear algebra. Always available."""

    name = "numpy"

    def abcd_to_s(self, abcd: np.ndarray, z0: np.ndarray) -> np.ndarray:
        Nf, N, _ = abcd.shape
        k = N // 2

        A = abcd[:, :k, :k]
        B = abcd[:, :k, k:]
        C = abcd[:, k:, :k]
        D = abcd[:, k:, k:]

        eye = np.broadcast_to(np.eye(k, dtype=complex), (Nf, k, k))
        Cinv_CinvD = np.linalg.solve(C, np.concatenate([eye, D], axis=-1))
        Cinv = Cinv_CinvD[:, :, :k]
        CinvD = Cinv_CinvD[:, :, k:]

        Z = np.empty((Nf, N, N), dtype=complex)
        Z[:, :k, :k] = A @ Cinv
        Z[:, :k, k:] = A @ Cinv @ D - B  # block form; A@Cinv@D != (A@D)@Cinv unless C,D commute
        Z[:, k:, :k] = Cinv
        Z[:, k:, k:] = CinvD

        # diagonal matrices  diag(Z0)  and  diag(conj(Z0))
        Z0d = np.zeros((Nf, N, N), dtype=complex)
        Z0cd = np.zeros((Nf, N, N), dtype=complex)
        idx = np.arange(N)
        Z0d[:, idx, idx] = z0
        Z0cd[:, idx, idx] = np.conj(z0)

        # power-wave S = √G0 (Z - Z0*) (Z + Z0)^{-1} √G0^{-1}
        G0 = np.real(z0)
        sqrtG0 = np.sqrt(G0)
        inv_sqrtG0 = 1.0 / sqrtG0

        M = Z - Z0cd  # numerator
        P = Z + Z0d  # denominator
        # right-solve P^{-1}: S0 = M P^{-1}  ->  P^T S0^T = M^T
        S0 = np.linalg.solve(P.swapaxes(-1, -2), M.swapaxes(-1, -2)).swapaxes(-1, -2)

        # apply the diagonal √G0 (·) √G0^{-1} as row/column scaling
        return sqrtG0[:, :, None] * S0 * inv_sqrtG0[:, None, :]

    def redheffer_star(self, s2: np.ndarray, s1: np.ndarray) -> np.ndarray:
        Nf, N, _ = s1.shape
        k = N // 2

        S1_11 = s1[:, :k, :k]
        S1_12 = s1[:, :k, k:]
        S1_21 = s1[:, k:, :k]
        S1_22 = s1[:, k:, k:]

        S2_11 = s2[:, :k, :k]
        S2_12 = s2[:, :k, k:]
        S2_21 = s2[:, k:, :k]
        S2_22 = s2[:, k:, k:]

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
