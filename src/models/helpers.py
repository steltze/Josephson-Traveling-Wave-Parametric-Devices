from dataclasses import dataclass
import numpy as np


@dataclass
class Immittance:
    """

        Parameters
    ----------
    Z0 : float
        Port impedance in Ohms. Default 50 Ω.
    """

    Z0: float = 50.0


#     # 3 JJs in series
# Series(JJ(L), JJ(L), JJ(L))          # == JJ(3L)

# # 3 JJs in parallel
# Parallel(Series(JJ(L), JJ(L), JJ(L)), Capacitor(C_g))        # == JJ(L/3)

# # Two-mode Δ shunt
# Parallel(Capacitor(C_g), Capacitor(2*C_i))

# # SNAIL (future — just add a SNAIL class with phi^3 potential)
# Series(SNAIL(L_small, L_large, n=3, flux=0.5))

import numpy as np


def band_coupling(n: int, offset: int) -> np.ndarray:
    """n x n symmetric matrix with 1s on the +/- `offset` diagonals.

    offset=1 -> nearest-neighbour band coupling (the B1 in L0(1+eps B1+...)).
    offset=2 -> second-neighbour coupling (the eps^2 'C' matrix), etc.
    """
    M = np.zeros((n, n))
    if abs(offset) >= n or offset == 0:
        return M
    idx = np.arange(n - abs(offset))
    M[idx, idx + abs(offset)] = 1.0
    M[idx + abs(offset), idx] = 1.0
    return M


def Ztot_parallel_LC(n, order, eps, L0, Ccap, Omega, coeffs):
    """Parallel inductor || capacitor in band space, via the M / A-inverse route.

    Derivation (matrix KCL, then your inverse expansion)
    ----------------------------------------------------
        L     = L0 ( I + sum_p eps^p coeffs[p] B_p ),   B_p = band_coupling(n, p)
        M     = j Omega L
        KCL   : ( j M Omega Ccap + I ) V = M I     ->   Z = A^{-1} M,
                A = j M Omega Ccap + I
        A^{-1}: A = A0 + sum_{t>=1} eps^t A_t, inverted by the consistent recursion
                X_0 = A0^{-1},  X_t = -A0^{-1} sum_{s=1}^{t} A_s X_{t-s}
        Z     = ( sum_t eps^t X_t ) ( sum_s eps^s M_s ), truncated at eps^order.

    A0 is diagonal with entries (1 - omega_k^2 L0 Ccap) — the PARALLEL tank
    denominators (omega^2 in the numerator of the correction, no 1/omega^2).

    Parameters
    ----------
    n       : number of bands (matrix dimension)
    order   : truncation order in eps (>= 0)
    eps     : perturbation amplitude (scalar)
    L0      : base inductance
    Ccap    : capacitance (scalar, constant) — the parallel capacitor
    Omega   : (n,n) diagonal matrix diag(omega_k), or 1D array of omega_k
    coeffs  : dict {p: amplitude}, e.g. {1: 1.0, 2: 1.0} for L0(1 + eps B1 + eps^2 B2)

    Returns
    -------
    (n, n) complex ndarray : Z_tot truncated at eps^order.

    Notes
    -----
    First-order perturbation theory degrades near the LC resonance
    omega_k = 1/sqrt(L0 Ccap), where A0 (hence Z) becomes singular.
    """
    Omega = np.diag(Omega) if np.ndim(Omega) == 1 else np.asarray(Omega)

    # --- M = j Omega L, grouped by power of eps:  M = sum_t eps^t Mc[t] ---
    Mc = {0: 1j * Omega * L0}
    for p, a in coeffs.items():
        if p <= order:
            Mc[p] = 1j * Omega @ (L0 * a * band_coupling(n, p))

    # --- A = j M Omega Ccap + I, grouped by power of eps:  A = sum_t eps^t A[t] ---
    A = {t: np.zeros((n, n), dtype=complex) for t in range(order + 1)}
    A[0] += np.eye(n, dtype=complex)
    for t, Mt in Mc.items():
        if t <= order:
            A[t] += 1j * Mt @ Omega * Ccap

    # --- A^{-1} consistent series:  X_0=A0^{-1}; X_t=-A0^{-1} sum A_s X_{t-s} ---
    A0inv = np.linalg.inv(A[0])
    X = {0: A0inv}
    for t in range(1, order + 1):
        acc = np.zeros((n, n), dtype=complex)
        for s in range(1, t + 1):
            acc += A[s] @ X[t - s]
        X[t] = -A0inv @ acc

    # --- Z = A^{-1} M, Cauchy product of the two eps-series, truncated ---
    Z = {t: np.zeros((n, n), dtype=complex) for t in range(order + 1)}
    for t in range(order + 1):
        for s in range(t + 1):
            if s in Mc:
                Z[t] += X[t - s] @ Mc[s]

    return sum((eps ** t) * Z[t] for t in range(order + 1))


def Zs_inductor(n, eps, L0, Omega, coeffs, order=None):
    """Bare series inductor M = j Omega L (no capacitor). Already a finite
    polynomial in eps; `order` (if given) drops band-couplings p > order."""
    Omega = np.diag(Omega) if np.ndim(Omega) == 1 else np.asarray(Omega)
    L = np.eye(n, dtype=complex)
    for p, a in coeffs.items():
        if order is None or p <= order:
            L = L + (eps ** p) * a * band_coupling(n, p)
    return 1j * Omega @ (L0 * L)


if __name__ == "__main__":
    # self-test: converges to the exact (non-truncated) inverse
    n = 4; L0 = 2.0; Ccap = 0.5
    w = np.array([0.3, 0.55, 0.72, 0.91]); Omega = np.diag(w); eps = 0.05
    coeffs = {1: 1.0, 2: 1.0}

    L = np.eye(n, dtype=complex)
    for p, a in coeffs.items():
        L += (eps ** p) * a * band_coupling(n, p)
    Yex = np.linalg.inv(1j * Omega @ (L0 * L)) + 1j * Omega * Ccap
    Zex = np.linalg.inv(Yex)

    for order in range(7):
        Zt = Ztot_parallel_LC(n, order, eps, L0, Ccap, Omega, coeffs)
        print(f"order={order}: max|Z_trunc - Z_exact| = {np.max(np.abs(Zt - Zex)):.3e}")