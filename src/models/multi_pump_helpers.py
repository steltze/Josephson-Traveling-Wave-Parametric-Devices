# """
# Two-pump sideband matrix construction for a Josephson TWPA/TWPC.

# Single pump:   omega_k   = omega_s + k*omega_p          (1D comb, index k)
# Two pumps:     omega_kn  = omega_s + k*omega_p1 + n*omega_p2   (2D lattice, index (k,n))

# Matrices:
#     omega_bar = diag(omega_kn)
#               = omega_s I + omega_p1 (K kron I_n) + omega_p2 (I_k kron N)

#     L_bar / L_J = I
#                 + eps1 (B1 kron I_n)          # pump-1 hop: k -> k+-1
#                 + eps2 (I_k kron B1)          # pump-2 hop: n -> n+-1
#                 + eps1^2 (B2 kron I_n)        # two pump-1 photons: k -> k+-2
#                 + eps2^2 (I_k kron B2)        # two pump-2 photons: n -> n+-2
#                 + eps1*eps2 (B1 kron B1)      # one photon from each: (k,n)->(k+-1,n+-1)
#                 + ...

# Index flattening convention: k is the OUTER (block) index, n is the INNER index,
# so the flat index is  idx = (k+Kmax)*(2N+1) + (n+Nmax).
# """

# import numpy as np


# def index_matrix(M):
#     """Diagonal matrix holding the sideband indices -M,...,-1,0,1,...,M."""
#     return np.diag(np.arange(-M, M + 1, dtype=float))


# def hop_matrix(M, step=1, theta=0.0):
#     """
#     Nearest/next-nearest hopping matrix on a (2M+1) sideband ladder.

#     step=1 -> B1 (couples k <-> k+-1), step=2 -> B2 (couples k <-> k+-2).
#     theta  -> pump phase: super-diagonal gets e^{+i theta}, sub-diagonal e^{-i theta}
#               (Hermitian). theta=0 gives the real symmetric matrix with +1 entries.
#     """
#     dim = 2 * M + 1
#     B = np.zeros((dim, dim), dtype=complex)
#     up = np.exp(1j * theta)
#     dn = np.exp(-1j * theta)
#     for i in range(dim - step):
#         B[i, i + step] = up      # super-diagonal: k -> k+step
#         B[i + step, i] = dn      # sub-diagonal:   k -> k-step
#     if theta == 0.0:
#         B = B.real.astype(float)
#     return B


# def build_single_pump(omega_s, omega_p, eps, Kmax, orders=1, theta=0.0):
#     """
#     Single-pump matrices.

#     Returns (omega_bar, L_over_LJ), each (2Kmax+1) square.
#     orders = highest power of eps to include (1 -> just B1, 2 -> B1 and B2, ...).
#     """
#     dimK = 2 * Kmax + 1
#     I = np.eye(dimK)
#     K = index_matrix(Kmax)

#     omega_bar = omega_s * I + omega_p * K

#     L = I.astype(complex) if theta != 0 else I.copy()
#     for m in range(1, orders + 1):
#         L = L + (eps ** m) * hop_matrix(Kmax, step=m, theta=theta)
#     return omega_bar, L


# def build_two_pump(omega_s, omega_p1, omega_p2, eps1, eps2,
#                    Kmax, Nmax, orders=1, theta1=0.0, theta2=0.0):
#     """
#     Two-pump matrices via Kronecker products.

#     Returns (omega_bar, L_over_LJ), each (2Kmax+1)(2Nmax+1) square.
#     orders=1 -> single-photon hops per pump + the eps1*eps2 cross term.
#     orders=2 -> also the eps1^2 and eps2^2 two-photon hops.
#     """
#     dimK = 2 * Kmax + 1
#     dimN = 2 * Nmax + 1
#     Ik = np.eye(dimK)
#     In = np.eye(dimN)
#     K = index_matrix(Kmax)
#     N = index_matrix(Nmax)

#     # --- frequency matrix ---
#     omega_bar = (omega_s * np.kron(Ik, In)
#                  + omega_p1 * np.kron(K, In)
#                  + omega_p2 * np.kron(Ik, N))

#     # --- inductance matrix ---
#     B1k = hop_matrix(Kmax, step=1, theta=theta1)
#     B1n = hop_matrix(Nmax, step=1, theta=theta2)

#     L = np.kron(Ik, In).astype(B1k.dtype)
#     # first-order single-pump hops
#     L = L + eps1 * np.kron(B1k, In.astype(B1k.dtype))
#     L = L + eps2 * np.kron(Ik.astype(B1n.dtype), B1n)
#     # cross term: one photon from each pump
#     L = L + (eps1 * eps2) * np.kron(B1k, B1n)

#     if orders >= 2:
#         B2k = hop_matrix(Kmax, step=2, theta=theta1)
#         B2n = hop_matrix(Nmax, step=2, theta=theta2)
#         L = L + (eps1 ** 2) * np.kron(B2k, In.astype(B2k.dtype))
#         L = L + (eps2 ** 2) * np.kron(Ik.astype(B2n.dtype), B2n)

#     return omega_bar, L


# def kn_labels(Kmax, Nmax):
#     """List of (k,n) pairs in the flattening order (k outer, n inner)."""
#     return [(k, n) for k in range(-Kmax, Kmax + 1)
#             for n in range(-Nmax, Nmax + 1)]


# if __name__ == "__main__":
#     np.set_printoptions(precision=2, suppress=True, linewidth=140)

#     # ---- demo parameters ----
#     omega_s = 6.0
#     omega_p1, omega_p2 = 12.0, 7.0
#     eps1, eps2 = 0.05, 0.03
#     Kmax, Nmax = 1, 1   # small so the matrices are readable

#     print("=== SINGLE PUMP (Kmax=2) ===")
#     ob1, L1 = build_single_pump(omega_s, omega_p1, eps1, Kmax=2, orders=2)
#     print("omega_bar diagonal:", np.diag(ob1))
#     print("L/L_J =\n", L1)

#     print("\n=== TWO PUMPS (Kmax=Nmax=1) ===")
#     ob, L = build_two_pump(omega_s, omega_p1, omega_p2, eps1, eps2,
#                            Kmax, Nmax, orders=2)
#     labels = kn_labels(Kmax, Nmax)
#     print("lattice (k,n) order:", labels)
#     print("omega_bar diagonal (frequencies at each (k,n)):")
#     for (k, n), w in zip(labels, np.diag(ob)):
#         print(f"   (k={k:+d}, n={n:+d}) -> omega = {w:6.2f}")

#     print("\nL/L_J (rows/cols in the (k,n) order above):")
#     print(np.real_if_close(L))

#     # sanity checks
#     print("\n=== checks ===")
#     print("omega_bar diagonal? ", np.allclose(ob, np.diag(np.diag(ob))))
#     print("L Hermitian (theta=0 -> symmetric)? ", np.allclose(L, L.T.conj()))
#     # verify a couple of couplings exist where expected
#     lab_index = {lab: i for i, lab in enumerate(labels)}
#     i0 = lab_index[(0, 0)]
#     ik = lab_index[(1, 0)]
#     inn = lab_index[(0, 1)]
#     ikn = lab_index[(1, 1)]
#     print(f"(0,0)->(1,0) coupling = {L[i0, ik].real:.3f}  (expect eps1={eps1})")
#     print(f"(0,0)->(0,1) coupling = {L[i0, inn].real:.3f}  (expect eps2={eps2})")
#     print(f"(0,0)->(1,1) coupling = {L[i0, ikn].real:.4f}  (expect eps1*eps2={eps1*eps2})")

"""
Multi-pump sideband matrix construction for a Josephson TWPA/TWPC.

For P pumps, idler frequencies live on a P-dimensional integer lattice:

    omega_{k_1,...,k_P} = omega_s + sum_j k_j * omega_{p,j}

Matrices (each of dimension  prod_j (2 Kmax_j + 1) ):

    omega_bar = omega_s * I_tot
              + sum_j omega_{p,j} * kron(I_1,...,K^{(j)},...,I_P)

    L_bar/L_J = I_tot
              + sum_j eps_j * kron(I_1,...,B1^{(j)},...,I_P)   # per-pump single hop
              + (cross terms generated by iteration; optional explicit pair terms)

This reduces to the single-pump case for P=1 and the two-pump case for P=2.
"""

import numpy as np
from functools import reduce


def kron_list(mats):
    """Kronecker product of a list of matrices, left to right."""
    return reduce(np.kron, mats)


def index_matrix(M):
    """diag(-M,...,-1,0,1,...,M)."""
    return np.diag(np.arange(-M, M + 1, dtype=float))


def hop_matrix(M, step=1, theta=0.0):
    """(2M+1) hopping matrix; super-diag e^{+i theta}, sub-diag e^{-i theta}."""
    dim = 2 * M + 1
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - step):
        B[i, i + step] = np.exp(1j * theta)
        B[i + step, i] = np.exp(-1j * theta)
    return B.real.astype(float) if theta == 0.0 else B


def build_multi_pump(omega_s, omega_p, eps, Kmax,
                     theta=None, pair_cross=True, orders=1):
    """
    Multi-pump matrices for P = len(omega_p) pumps.

    Parameters
    ----------
    omega_s : float                signal frequency
    omega_p : sequence of P floats  pump frequencies
    eps     : sequence of P floats  modulation depths
    Kmax    : sequence of P ints    sideband truncation per pump
    theta   : sequence of P floats  pump phases (default all 0)
    pair_cross : bool               include explicit eps_i*eps_j pair hops
    orders  : int                   per-pump highest harmonic (1->B1, 2->+B2, ...)

    Returns
    -------
    omega_bar, L_over_LJ : (D x D) arrays, D = prod_j (2 Kmax_j + 1)
    """
    P = len(omega_p)
    assert len(eps) == P and len(Kmax) == P
    if theta is None:
        theta = [0.0] * P

    dims = [2 * K + 1 for K in Kmax]
    Is = [np.eye(d) for d in dims]
    D = int(np.prod(dims))

    # ---------- frequency matrix ----------
    omega_bar = omega_s * np.eye(D)
    for j in range(P):
        mats = list(Is)
        mats[j] = index_matrix(Kmax[j])
        omega_bar = omega_bar + omega_p[j] * kron_list(mats)

    # ---------- inductance matrix ----------
    # dtype: complex if any phase nonzero
    cdtype = complex if any(t != 0.0 for t in theta) else float
    L = np.eye(D).astype(cdtype)

    # per-pump single hops (and higher harmonics if orders>1)
    B1s = [hop_matrix(Kmax[j], step=1, theta=theta[j]) for j in range(P)]
    for j in range(P):
        for m in range(1, orders + 1):
            mats = [I.astype(cdtype) for I in Is]
            mats[j] = hop_matrix(Kmax[j], step=m, theta=theta[j]).astype(cdtype)
            L = L + (eps[j] ** m) * kron_list(mats)

    # explicit pairwise cross terms eps_i eps_j (hop two indices at once)
    if pair_cross:
        for i in range(P):
            for j in range(i + 1, P):
                mats = [I.astype(cdtype) for I in Is]
                mats[i] = B1s[i].astype(cdtype)
                mats[j] = B1s[j].astype(cdtype)
                L = L + (eps[i] * eps[j]) * kron_list(mats)

    return omega_bar, L


def lattice_labels(Kmax):
    """All index tuples (k_1,...,k_P) in the kron flattening order."""
    from itertools import product
    ranges = [range(-K, K + 1) for K in Kmax]
    return list(product(*ranges))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=160)

    # ----- P = 1 (recovers single pump) -----
    ob, L = build_multi_pump(6.0, [12.0], [0.05], [2], orders=2)
    print("P=1  omega diag:", np.diag(ob))
    print("P=1  L/L_J:\n", L, "\n")

    # ----- P = 3 pumps -----
    omega_s = 6.0
    omega_p = [12.0, 7.0, 5.0]
    eps     = [0.05, 0.03, 0.02]
    Kmax    = [1, 1, 1]           # 3*3*3 = 27 lattice nodes

    ob, L = build_multi_pump(omega_s, omega_p, eps, Kmax)
    labels = lattice_labels(Kmax)
    D = len(labels)
    print(f"P=3  lattice dimension D = {D}")
    print("sample frequencies at a few nodes:")
    for tup in [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,-1,0),(1,1,-1)]:
        idx = labels.index(tup)
        print(f"   k={tup} -> omega = {np.diag(ob)[idx]:6.2f}")

    # verify some couplings
    lut = {t: i for i, t in enumerate(labels)}
    print("\ncoupling checks:")
    print(f"  (0,0,0)->(1,0,0) = {L[lut[(0,0,0)], lut[(1,0,0)]]:.3f}  (eps1=0.05)")
    print(f"  (0,0,0)->(0,1,0) = {L[lut[(0,0,0)], lut[(0,1,0)]]:.3f}  (eps2=0.03)")
    print(f"  (0,0,0)->(0,0,1) = {L[lut[(0,0,0)], lut[(0,0,1)]]:.3f}  (eps3=0.02)")
    print(f"  (0,0,0)->(1,1,0) = {L[lut[(0,0,0)], lut[(1,1,0)]]:.4f}  (eps1*eps2={0.05*0.03})")
    print(f"  omega_bar diagonal? {np.allclose(ob, np.diag(np.diag(ob)))}")
    print(f"  L symmetric?        {np.allclose(L, L.T)}")

    # dimension growth illustration
    print("\nstate dimension vs number of pumps (5 sidebands each, Kmax=2):")
    for Pn in range(1, 6):
        print(f"   P={Pn}:  D = {5**Pn}")