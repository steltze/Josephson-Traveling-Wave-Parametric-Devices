from __future__ import annotations

import numpy as np

from backends.base import Backend


class NumpyBackend(Backend):
    """Reference backend using NumPy's batched linear algebra. Always available."""

    name = "numpy"

    def single_mode_matrix(self, Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
        Nf, m, _ = Zs.shape
        I = np.eye(m, dtype=complex)

        T = np.empty((Nf, 2 * m, 2 * m), dtype=complex)
        T[:, :m, :m] = I
        T[:, :m, m:] = -Zs
        T[:, m:, :m] = -Yg
        T[:, m:, m:] = I + Yg @ Zs
        return T

    def symmetric_single_mode_matrix(
        self, Zs: np.ndarray, Yg: np.ndarray
    ) -> np.ndarray:

        Nf, m, _ = Zs.shape
        I = np.eye(m, dtype=complex)
        Yg_half = Yg / 2
        T = np.empty((Nf, 2 * m, 2 * m), dtype=complex)
        T[:, :m, :m] = I + Zs @ Yg_half
        T[:, :m, m:] = -Zs
        T[:, m:, :m] = -Yg - Yg_half @ Zs @ Yg_half
        T[:, m:, m:] = I + Yg_half @ Zs
        return T

    def slot_mode_matrix(
        self,
        Zs: np.ndarray,
        Yg: np.ndarray,
        Zs_slot: np.ndarray,
        Yg_slot: np.ndarray,
        Yi_coupling: np.ndarray,
    ) -> np.ndarray:
        """
        Asymmetric slot-mode cell: main JTL line (series Zs, shunt-to-ground
        Yg) running alongside a slot line (series Zs_slot, shunt-to-ground
        Yg_slot), the two coupled node-to-node by a shunt admittance
        Yi_coupling between them.

        State per cell: x = [V', V, I_s, I] (each an m-vector of harmonics),
        V'/I_s the slot line and V/I the main line. Current sign convention
        matches `single_mode_matrix`: I_n (I_s,n) is the current flowing
        into cell n from the n+1 side, so setting Yi_coupling = 0 collapses
        rows 1&3 and 2&4 each to a plain, independent `single_mode_matrix`
        L-cell (Zs_slot,Yg_slot) / (Zs,Yg).

        Derivation (KVL on each series branch, then KCL at each shunt node,
        substituting V'_{n+1}, V_{n+1} back out via the KVL rows so every
        row is expressed in x_{n+1} alone) gives det(T) = 1 exactly for any
        reactive Zs, Zs_slot, Yg, Yg_slot, Yi_coupling, as required for a
        lossless reciprocal cell.

        Cell size: 4m x 4m, batched over Nf frequencies.

        Parameters
        ----------
        Zs, Yg, Zs_slot, Yg_slot, Yi_coupling : ndarray, complex, shape (Nf, m, m)
        """
        Nf, m, _ = Zs.shape
        I = np.eye(m, dtype=complex)
        T = np.empty((Nf, 4 * m, 4 * m), dtype=complex)
        Zeros = np.zeros((m, m), dtype=complex)

        T[:, 0 * m : 1 * m, 0 * m : 1 * m] = I
        T[:, 0 * m : 1 * m, 1 * m : 2 * m] = Zeros
        T[:, 0 * m : 1 * m, 2 * m : 3 * m] = -Zs_slot
        T[:, 0 * m : 1 * m, 3 * m : 4 * m] = Zeros

        T[:, 1 * m : 2 * m, 0 * m : 1 * m] = Zeros
        T[:, 1 * m : 2 * m, 1 * m : 2 * m] = I
        T[:, 1 * m : 2 * m, 2 * m : 3 * m] = Zeros
        T[:, 1 * m : 2 * m, 3 * m : 4 * m] = -Zs

        T[:, 2 * m : 3 * m, 0 * m : 1 * m] = -(Yg_slot + Yi_coupling)
        T[:, 2 * m : 3 * m, 1 * m : 2 * m] = Yi_coupling
        T[:, 2 * m : 3 * m, 2 * m : 3 * m] = I + (Yg_slot + Yi_coupling) @ Zs_slot
        T[:, 2 * m : 3 * m, 3 * m : 4 * m] = -Yi_coupling @ Zs

        T[:, 3 * m : 4 * m, 0 * m : 1 * m] = Yi_coupling
        T[:, 3 * m : 4 * m, 1 * m : 2 * m] = -(Yg + Yi_coupling)
        T[:, 3 * m : 4 * m, 2 * m : 3 * m] = -Yi_coupling @ Zs_slot
        T[:, 3 * m : 4 * m, 3 * m : 4 * m] = I + (Yg + Yi_coupling) @ Zs
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

        M = np.empty((Nf, N, N), dtype=complex)
        M[:, :k, :k] = A @ Cinv
        # A@Cinv@D != (A@D)@Cinv unless C,D commute, so this must go through
        # Cinv first
        M[:, :k, k:] = M[:, :k, :k] @ D - B
        M[:, k:, :k] = Cinv
        M[:, k:, k:] = CinvD
        idx = np.arange(N)
        M[:, idx, idx] -= np.conj(z0)

        G0 = np.real(z0)
        sqrtG0 = 1 / np.sqrt(G0)
        inv_sqrtG0 = 1.0 / sqrtG0

        P = M.copy()
        P[:, idx, idx] += 2.0 * G0

        S0 = np.linalg.solve(P.swapaxes(-1, -2), M.swapaxes(-1, -2)).swapaxes(-1, -2)

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

        A1 = np.eye(k)[None] - S2_11 @ S1_22
        X1 = np.linalg.solve(A1, np.concatenate([S2_11 @ S1_21, S2_12], axis=-1))

        A2 = np.eye(k)[None] - S1_22 @ S2_11
        X2 = np.linalg.solve(A2, np.concatenate([S1_21, S1_22 @ S2_12], axis=-1))

        S = np.empty((Nf, N, N), dtype=complex)
        S[:, :k, :k] = S1_11 + S1_12 @ X1[:, :, :k]
        S[:, :k, k:] = S1_12 @ X1[:, :, k:]
        S[:, k:, :k] = S2_21 @ X2[:, :, :k]
        S[:, k:, k:] = S2_22 + S2_21 @ X2[:, :, k:]

        return S
