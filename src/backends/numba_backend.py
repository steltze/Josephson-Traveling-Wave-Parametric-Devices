from __future__ import annotations

import numpy as np

from backends.base import Backend

try:
    import numba
except ImportError as exc:  # surfaced by registry.get_backend as a chained ImportError
    raise ImportError(
        "The 'numba' backend requires the numba package. Install it with "
        "`pip install .[numba]`."
    ) from exc


@numba.njit(cache=True)
def _abcd_to_s_single(abcd: np.ndarray, z0: np.ndarray) -> np.ndarray:
    N = abcd.shape[0]
    k = N // 2

    A = np.ascontiguousarray(abcd[:k, :k])
    B = np.ascontiguousarray(abcd[:k, k:])
    C = np.ascontiguousarray(abcd[k:, :k])
    D = np.ascontiguousarray(abcd[k:, k:])

    eye = np.eye(k, dtype=np.complex128)
    Cinv_CinvD = np.linalg.solve(C, np.concatenate((eye, D), axis=1))
    Cinv = np.ascontiguousarray(Cinv_CinvD[:, :k])
    CinvD = np.ascontiguousarray(Cinv_CinvD[:, k:])

    Z = np.empty((N, N), dtype=np.complex128)
    Z[:k, :k] = A @ Cinv
    Z[:k, k:] = A @ Cinv @ D - B  # block form; A@Cinv@D != (A@D)@Cinv unless C,D commute
    Z[k:, :k] = Cinv
    Z[k:, k:] = CinvD

    Z0d = np.zeros((N, N), dtype=np.complex128)
    Z0cd = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        Z0d[i, i] = z0[i]
        Z0cd[i, i] = np.conj(z0[i])

    G0 = np.real(z0)
    sqrtG0 = np.sqrt(G0)
    inv_sqrtG0 = 1.0 / sqrtG0

    X = np.linalg.solve((Z + Z0d).T, (Z - Z0cd).T)
    S0 = X.T

    S = np.empty((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            S[i, j] = sqrtG0[i] * S0[i, j] * inv_sqrtG0[j]
    return S


@numba.njit(parallel=True, cache=True)
def _abcd_to_s_batch(abcd: np.ndarray, z0: np.ndarray) -> np.ndarray:
    Nf, N, _ = abcd.shape
    S = np.empty((Nf, N, N), dtype=np.complex128)
    for f in numba.prange(Nf):
        S[f] = _abcd_to_s_single(abcd[f], z0[f])
    return S


@numba.njit(cache=True)
def _redheffer_star_single(s2: np.ndarray, s1: np.ndarray) -> np.ndarray:
    N = s1.shape[0]
    k = N // 2

    S1_11 = np.ascontiguousarray(s1[:k, :k])
    S1_12 = np.ascontiguousarray(s1[:k, k:])
    S1_21 = np.ascontiguousarray(s1[k:, :k])
    S1_22 = np.ascontiguousarray(s1[k:, k:])

    S2_11 = np.ascontiguousarray(s2[:k, :k])
    S2_12 = np.ascontiguousarray(s2[:k, k:])
    S2_21 = np.ascontiguousarray(s2[k:, :k])
    S2_22 = np.ascontiguousarray(s2[k:, k:])

    eye_k = np.eye(k, dtype=np.complex128)

    A1 = eye_k - S2_11 @ S1_22
    X1 = np.linalg.solve(A1, np.concatenate((S2_11 @ S1_21, S2_12), axis=1))
    X1_left = np.ascontiguousarray(X1[:, :k])
    X1_right = np.ascontiguousarray(X1[:, k:])

    A2 = eye_k - S1_22 @ S2_11
    X2 = np.linalg.solve(A2, np.concatenate((S1_21, S1_22 @ S2_12), axis=1))
    X2_left = np.ascontiguousarray(X2[:, :k])
    X2_right = np.ascontiguousarray(X2[:, k:])

    S = np.empty((N, N), dtype=np.complex128)
    S[:k, :k] = S1_11 + S1_12 @ X1_left
    S[:k, k:] = S1_12 @ X1_right
    S[k:, :k] = S2_21 @ X2_left
    S[k:, k:] = S2_22 + S2_21 @ X2_right
    return S


@numba.njit(parallel=True, cache=True)
def _redheffer_star_batch(s2: np.ndarray, s1: np.ndarray) -> np.ndarray:
    Nf, N, _ = s1.shape
    S = np.empty((Nf, N, N), dtype=np.complex128)
    for f in numba.prange(Nf):
        S[f] = _redheffer_star_single(s2[f], s1[f])
    return S


@numba.njit(parallel=True, cache=True)
def _cascade_all_batch(s_cells: np.ndarray) -> np.ndarray:
    """Reduce all Nc cells per frequency inside one compiled call — no
    per-cell Python-level dispatch, unlike calling redheffer_star Nc times."""
    Nf, Nc, N, _ = s_cells.shape
    S = np.empty((Nf, N, N), dtype=np.complex128)
    for f in numba.prange(Nf):
        total = s_cells[f, 0]
        for c in range(1, Nc):
            total = _redheffer_star_single(s_cells[f, c], total)
        S[f] = total
    return S


class NumbaBackend(Backend):
    """
    JIT-compiled, multi-threaded backend (Numba `@njit(parallel=True)`).

    Same numerics as `NumpyBackend`, executed by compiling an explicit
    per-frequency kernel and running it across the frequency axis with
    `numba.prange`. First call per input shape/dtype pays a JIT
    compilation cost.

    Measured behaviour (see `cascade_all`'s docstring for the benchmark
    setup): this backend wins over NumpyBackend as the port count N grows
    (more Floquet sidebands — larger M / longer ks_state), and loses for
    small N (e.g. N=4, the common single-idler case) even after fusing
    the whole cell cascade into one call. For small N the bottleneck is
    not Python dispatch but the fixed per-call cost of solving many tiny
    linear systems — fusing removes the former but not the latter.

    Requires the optional `numba` dependency: `pip install .[numba]`.
    """

    name = "numba"

    def abcd_to_s(self, abcd: np.ndarray, z0: np.ndarray) -> np.ndarray:
        return _abcd_to_s_batch(np.ascontiguousarray(abcd), np.ascontiguousarray(z0))

    def redheffer_star(self, s2: np.ndarray, s1: np.ndarray) -> np.ndarray:
        return _redheffer_star_batch(np.ascontiguousarray(s2), np.ascontiguousarray(s1))

    def cascade_all(self, s_cells: np.ndarray) -> np.ndarray:
        """
        Fused cascade: the whole per-cell reduction compiles to one kernel
        (single dispatch for all Nc cells, vs Nc calls into `redheffer_star`
        from Python for the default `Backend.cascade_all`).

        Measured on a 321-cell, 500-frequency-point cascade: this closes
        essentially none of the gap at N=4 (fused: still ~2x slower than
        NumPy) — the cost there is dominated by the per-call overhead of
        `np.linalg.solve` on many tiny (k=2) systems, not by Python
        dispatch, so fusing the outer loop doesn't touch it. It matters
        more as N grows: ~1.6x faster than NumPy at N=12, ~2.4x at N=16,
        where each per-frequency solve is large enough for that per-call
        overhead to stop dominating.
        """
        return _cascade_all_batch(np.ascontiguousarray(s_cells))
