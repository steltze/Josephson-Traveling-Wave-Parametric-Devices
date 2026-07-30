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
def _single_mode_matrix_single(Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
    m = Zs.shape[0]
    I = np.eye(m, dtype=np.complex128)
    T = np.empty((2 * m, 2 * m), dtype=np.complex128)
    T[:m, :m] = I
    T[:m, m:] = -Zs
    T[m:, :m] = -Yg
    T[m:, m:] = I + Yg @ Zs
    return T


@numba.njit(parallel=True, cache=True)
def _single_mode_matrix_batch(Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
    Nf, m, _ = Zs.shape
    T = np.empty((Nf, 2 * m, 2 * m), dtype=np.complex128)
    for f in numba.prange(Nf):
        T[f] = _single_mode_matrix_single(Zs[f], Yg[f])
    return T


@numba.njit(cache=True)
def _symmetric_single_mode_matrix_single(Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
    m = Zs.shape[0]
    I = np.eye(m, dtype=np.complex128)
    Yg_half = Yg / 2
    T = np.empty((2 * m, 2 * m), dtype=np.complex128)
    T[:m, :m] = I + Zs @ Yg_half
    T[:m, m:] = -Zs
    T[m:, :m] = -Yg - Yg_half @ Zs @ Yg_half
    T[m:, m:] = I + Yg_half @ Zs
    return T


@numba.njit(parallel=True, cache=True)
def _symmetric_single_mode_matrix_batch(Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
    Nf, m, _ = Zs.shape
    T = np.empty((Nf, 2 * m, 2 * m), dtype=np.complex128)
    for f in numba.prange(Nf):
        T[f] = _symmetric_single_mode_matrix_single(Zs[f], Yg[f])
    return T


@numba.njit(parallel=True, cache=True)
def _single_mode_matrix_grid_batch(Zs_stack: np.ndarray, Yg_stack: np.ndarray, is_L: bool) -> np.ndarray:
    """Fused over both frequency and cell axes -- see `NumbaBackend.single_mode_matrix_grid`."""
    Nf, Nc, m, _ = Zs_stack.shape
    T = np.empty((Nf, Nc, 2 * m, 2 * m), dtype=np.complex128)
    for f in numba.prange(Nf):
        for c in range(Nc):
            if is_L:
                T[f, c] = _single_mode_matrix_single(Zs_stack[f, c], Yg_stack[f, c])
            else:
                T[f, c] = _symmetric_single_mode_matrix_single(Zs_stack[f, c], Yg_stack[f, c])
    return T


@numba.njit(cache=True)
def _slot_mode_matrix_single(
    Zs: np.ndarray,
    Yg: np.ndarray,
    Zs_slot: np.ndarray,
    Yg_slot: np.ndarray,
    Yi_coupling: np.ndarray,
) -> np.ndarray:
    m = Zs.shape[0]
    I = np.eye(m, dtype=np.complex128)
    Zeros = np.zeros((m, m), dtype=np.complex128)
    T = np.empty((4 * m, 4 * m), dtype=np.complex128)
    # ---- row 1 : V'_n = V'_{n+1} - Zs_slot @ I_s_{n+1} ----
    T[0 * m:1 * m, 0 * m:1 * m] = I
    T[0 * m:1 * m, 1 * m:2 * m] = Zeros
    T[0 * m:1 * m, 2 * m:3 * m] = -Zs_slot
    T[0 * m:1 * m, 3 * m:4 * m] = Zeros
    # ---- row 2 : V_n = V_{n+1} - Zs @ I_{n+1} ----
    T[1 * m:2 * m, 0 * m:1 * m] = Zeros
    T[1 * m:2 * m, 1 * m:2 * m] = I
    T[1 * m:2 * m, 2 * m:3 * m] = Zeros
    T[1 * m:2 * m, 3 * m:4 * m] = -Zs
    # ---- row 3 : I_s update (KCL at the slot node) ----
    T[2 * m:3 * m, 0 * m:1 * m] = -(Yg_slot + Yi_coupling)
    T[2 * m:3 * m, 1 * m:2 * m] = Yi_coupling
    T[2 * m:3 * m, 2 * m:3 * m] = I + (Yg_slot + Yi_coupling) @ Zs_slot
    T[2 * m:3 * m, 3 * m:4 * m] = -Yi_coupling @ Zs
    # ---- row 4 : I update (KCL at the main node) ----
    T[3 * m:4 * m, 0 * m:1 * m] = Yi_coupling
    T[3 * m:4 * m, 1 * m:2 * m] = -(Yg + Yi_coupling)
    T[3 * m:4 * m, 2 * m:3 * m] = -Yi_coupling @ Zs_slot
    T[3 * m:4 * m, 3 * m:4 * m] = I + (Yg + Yi_coupling) @ Zs
    return T


@numba.njit(parallel=True, cache=True)
def _slot_mode_matrix_batch(
    Zs: np.ndarray,
    Yg: np.ndarray,
    Zs_slot: np.ndarray,
    Yg_slot: np.ndarray,
    Yi_coupling: np.ndarray,
) -> np.ndarray:
    Nf, m, _ = Zs.shape
    T = np.empty((Nf, 4 * m, 4 * m), dtype=np.complex128)
    for f in numba.prange(Nf):
        T[f] = _slot_mode_matrix_single(Zs[f], Yg[f], Zs_slot[f], Yg_slot[f], Yi_coupling[f])
    return T


@numba.njit(parallel=True, cache=True)
def _slot_mode_matrix_grid_batch(
    Zs_stack: np.ndarray,
    Yg_stack: np.ndarray,
    Zs_slot_stack: np.ndarray,
    Yg_slot_stack: np.ndarray,
    Yi_coupling_stack: np.ndarray,
) -> np.ndarray:
    """Fused over both frequency and cell axes -- see `NumbaBackend.single_mode_matrix_grid`."""
    Nf, Nc, m, _ = Zs_stack.shape
    T = np.empty((Nf, Nc, 4 * m, 4 * m), dtype=np.complex128)
    for f in numba.prange(Nf):
        for c in range(Nc):
            T[f, c] = _slot_mode_matrix_single(
                Zs_stack[f, c],
                Yg_stack[f, c],
                Zs_slot_stack[f, c],
                Yg_slot_stack[f, c],
                Yi_coupling_stack[f, c],
            )
    return T


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

    # M = Z - diag(conj(z0)), applied as an in-place diagonal update instead
    # of allocating dense (N,N) diag(z0)/diag(conj(z0)) matrices just to
    # hold N diagonal values -- this kernel runs once per frequency across
    # `numba.prange` threads, so the saved allocations multiply by Nf.
    M = np.empty((N, N), dtype=np.complex128)
    M[:k, :k] = A @ Cinv
    M[:k, k:] = A @ Cinv @ D - B  # block form; A@Cinv@D != (A@D)@Cinv unless C,D commute
    M[k:, :k] = Cinv
    M[k:, k:] = CinvD

    G0 = np.real(z0)
    sqrtG0 = np.sqrt(G0)
    inv_sqrtG0 = 1.0 / sqrtG0

    for i in range(N):
        M[i, i] -= np.conj(z0[i])  # M is now Z - diag(conj(z0))
    P = M.copy()
    for i in range(N):
        P[i, i] += 2.0 * G0[i]  # P = M + diag(z0 + conj(z0)) = Z + diag(z0)

    X = np.linalg.solve(P.T, M.T)
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

    def single_mode_matrix(self, Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
        return _single_mode_matrix_batch(np.ascontiguousarray(Zs), np.ascontiguousarray(Yg))

    def symmetric_single_mode_matrix(self, Zs: np.ndarray, Yg: np.ndarray) -> np.ndarray:
        return _symmetric_single_mode_matrix_batch(
            np.ascontiguousarray(Zs), np.ascontiguousarray(Yg)
        )

    def slot_mode_matrix(
        self,
        Zs: np.ndarray,
        Yg: np.ndarray,
        Zs_slot: np.ndarray,
        Yg_slot: np.ndarray,
        Yi_coupling: np.ndarray,
    ) -> np.ndarray:
        return _slot_mode_matrix_batch(
            np.ascontiguousarray(Zs),
            np.ascontiguousarray(Yg),
            np.ascontiguousarray(Zs_slot),
            np.ascontiguousarray(Yg_slot),
            np.ascontiguousarray(Yi_coupling),
        )

    def single_mode_matrix_grid(self, cells: list, topology: str) -> np.ndarray:
        """
        Fused cascade-wide build: the whole (Nf, Ncells) loop compiles to
        one kernel (single dispatch, vs Ncells separate calls from Python
        for the default `Backend.single_mode_matrix_grid`) -- same
        rationale as `cascade_all` below.
        """
        if cells[0].Zs_slot_fn is not None:
            Zs_slot_stack = np.ascontiguousarray(
                np.stack([cell.Zs_slot_fn for cell in cells], axis=1)
            )
            Yg_slot_stack = np.ascontiguousarray(
                np.stack([cell.Yg_slot_fn for cell in cells], axis=1)
            )
            Yi_coupling_stack = np.ascontiguousarray(
                np.stack([cell.Yi_slot_coupling_fn for cell in cells], axis=1)
            )
            Zs_stack = np.ascontiguousarray(
                np.stack([cell.Zs_harm_fn for cell in cells], axis=1)
            )
            Yg_stack = np.ascontiguousarray(
                np.stack([cell.Yg_harm_fn for cell in cells], axis=1)
            )
            return _slot_mode_matrix_grid_batch(
                Zs_stack, Yg_stack, Zs_slot_stack, Yg_slot_stack, Yi_coupling_stack
            )

        Zs_stack = np.ascontiguousarray(
            np.stack([cell.Zs_harm_fn for cell in cells], axis=1)
        )
        Yg_stack = np.ascontiguousarray(
            np.stack([cell.Yg_harm_fn for cell in cells], axis=1)
        )
        return _single_mode_matrix_grid_batch(Zs_stack, Yg_stack, topology == "L")

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
