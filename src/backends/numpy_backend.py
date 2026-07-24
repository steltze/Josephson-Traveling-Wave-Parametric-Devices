from __future__ import annotations

import numpy as np

from backends.base import Backend


class NumpyBackend(Backend):
    """Reference backend using NumPy's batched linear algebra. Always available."""

    name = "numpy"

    def single_mode_matrix(self, Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
        Nf, m, _ = Zs.shape
        I = np.eye(m, dtype=complex)
        # Write blocks directly into the output instead of concatenate-ing
        # halves together -- avoids allocating `top`/`bottom` plus a third
        # full-size copy for the final join.
        T = np.empty((Nf, 2 * m, 2 * m), dtype=complex)
        T[:, :m, :m] = I
        T[:, :m, m:] = -Zs
        T[:, m:, :m] = -Yg
        T[:, m:, m:] = I + Yg @ Zs
        return T

    def symmetric_single_mode_matrix(self, Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
        # Symmetric pi-cell: shunt Yg/2 - series Zs - shunt Yg/2
        Nf, m, _ = Zs.shape
        I = np.eye(m, dtype=complex)
        Yg_half = Yg / 2
        T = np.empty((Nf, 2 * m, 2 * m), dtype=complex)
        T[:, :m, :m] = I + Zs @ Yg_half
        T[:, :m, m:] = -Zs
        T[:, m:, :m] = -Yg - Yg_half @ Zs @ Yg_half
        T[:, m:, m:] = I + Yg_half @ Zs
        return T

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

        # M = Z - diag(conj(Z0)), built directly rather than via a separate
        # Z plus a dense (Nf,N,N) diag(conj(Z0)) matrix -- Z0 only ever
        # touches N of the N^2 entries, so it's applied as an in-place
        # diagonal update instead of materializing it.
        M = np.empty((Nf, N, N), dtype=complex)
        M[:, :k, :k] = A @ Cinv
        # A@Cinv@D != (A@D)@Cinv unless C,D commute, so this must go through
        # Cinv first -- but that product is already sitting in M[:,:k,:k],
        # so reuse it instead of recomputing A@Cinv a second time.
        M[:, :k, k:] = M[:, :k, :k] @ D - B
        M[:, k:, :k] = Cinv
        M[:, k:, k:] = CinvD
        idx = np.arange(N)
        M[:, idx, idx] -= np.conj(z0)

        # power-wave S = √G0 (Z - Z0*) (Z + Z0)^{-1} √G0^{-1}
        G0 = np.real(z0)
        sqrtG0 = np.sqrt(G0)
        inv_sqrtG0 = 1.0 / sqrtG0

        # P = Z + diag(Z0) = M + diag(Z0 + conj(Z0)) = M + diag(2 Re(Z0))
        P = M.copy()
        P[:, idx, idx] += 2.0 * G0
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
