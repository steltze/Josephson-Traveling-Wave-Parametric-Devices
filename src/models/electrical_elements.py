from __future__ import annotations
from functools import reduce
import numpy as np


def band_coupling_phased(n: int, offset: int, theta: float) -> np.ndarray:
    """n x n complex band-coupling matrix for a pump modulation cos(w_p t - theta).

    Entry [target, source]: a target sideband above its source (up-shift,
    towards +w_p) picks up exp(-i*offset*theta); a target below its source
    (down-shift) picks up the conjugate exp(+i*offset*theta). Must stay a conjugate
    pair (not the same phase on both diagonals) or the sideband-coupling
    matrix stops being passive/lossless (fails Manley-Rowe / pseudo-unitarity).
    """
    M = np.zeros((n, n), dtype=complex)
    if abs(offset) >= n or offset == 0:
        return M
    idx = np.arange(n - abs(offset))
    M[idx, idx + abs(offset)] = 0.5*np.exp(
        1j * abs(offset) * theta
    )  # target < source: down-shift
    M[idx + abs(offset), idx] = 0.5*np.exp(
        -1j * abs(offset) * theta
    )  # target > source: up-shift
    return M


def _is_matrix(value: np.ndarray) -> bool:
    """True for a genuine (..., n, n) sideband-coupling matrix, as opposed
    to a per-sideband array of independent scalar values. Squareness alone
    isn't enough: a per-sideband array on a batched (Nf, n) grid also has
    ndim 2, and is square too whenever Nf happens to equal n. A genuine
    coupling matrix carries the sideband axis *in addition to* the Nf
    batch axis (Nf, n, n) -- ndim > 2, not just square, is what actually
    distinguishes the two."""
    return value.ndim == 3 and value.shape[-1] == value.shape[-2]


def _invert(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if _is_matrix(value):
        return np.linalg.inv(value)
    return 1.0 / value


def _diag_promote(value: np.ndarray) -> np.ndarray:
    """(..., n) -> (..., n, n) with `value`'s entries placed on the
    diagonal. Used to lift a plain component's per-sideband values (one
    impedance per sideband frequency) into the same matrix shape as a
    sibling ModulatedInductor, so they can be summed."""
    n = value.shape[-1]
    out = np.zeros(value.shape + (n,), dtype=complex)
    idx = np.arange(n)
    out[..., idx, idx] = value
    return out


def _sum_immittances(values: list[np.ndarray]) -> np.ndarray:
    values = [np.asarray(v) for v in values]
    matrices = [v for v in values if _is_matrix(v)]
    if not matrices:
        return sum(values)

    ns = {m.shape[-1] for m in matrices}
    if len(ns) > 1:
        raise ValueError(f"sideband dimensions don't match across components: {ns}")

    total = np.zeros(matrices[0].shape, dtype=complex)
    for v in values:
        total = total + (v if _is_matrix(v) else _diag_promote(v))
    return total


class Component:
    """Base class for two-terminal circuit elements.

    A value is a float/complex (or an array over frequency) for plain
    elements, or an (n, n) sideband-coupling matrix (or (Nf, n, n) when
    batched over frequency) for elements modulated by a pump tone.
    impedance/admittance are duals of each other via matrix/scalar inversion.
    """

    def impedance(self, omega):
        return _invert(self.admittance(omega))

    def admittance(self, omega):
        return _invert(self.impedance(omega))

    def impedance_matrix(self, omega):
        """impedance(omega) as an (..., n, n) matrix, diagonal if this
        component doesn't couple sidebands."""
        Z = np.asarray(self.impedance(omega))
        return Z if _is_matrix(Z) else _diag_promote(Z)

    def admittance_matrix(self, omega):
        """admittance(omega) as an (..., n, n) matrix, diagonal if this
        component doesn't couple sidebands."""
        Y = np.asarray(self.admittance(omega))
        return Y if _is_matrix(Y) else _diag_promote(Y)

    @staticmethod
    def series(*components: "Component") -> "Component":
        return _Combination(components, "series")

    @staticmethod
    def parallel(*components: "Component") -> "Component":
        return _Combination(components, "parallel")


class Inductor(Component):
    def __init__(self, L: float):
        self.L = L

    def impedance(self, omega):
        return 1j * np.asarray(omega) * self.L


class Capacitor(Component):
    def __init__(self, C: float):
        self.C = C

    def impedance(self, omega):
        return 1.0 / (1j * np.asarray(omega) * self.C)


class ModulatedInductor(Component):
    """
    Pump-modulated (Josephson) inductor, specified by the ABSOLUTE inverse-
    inductance harmonics.

        Linv(t) = sum_{p>=0} c[p] * cos(p*theta)   [1/henry]

    so the inverse-inductance matrix over sidebands is

        Linv_mat = c[0]*I + sum_{p>=1} c[p] * band_coupling_phased(n, p, theta)

    and the series impedance is  Z = j*Omega * L = j*Omega * inv(Linv_mat).

    `coeffs` must include the DC term as coeffs[0] (the identity coefficient),
    and coeffs[p] for p>=1 are the harmonic couplings. All have units 1/H.
    """

    def __init__(self, coeffs, theta=0.0, order=None):
        self.theta = theta
        self.coeffs = coeffs           # {0: c0, 1: c1, ...} in 1/H
        self.order = order if order is not None else max(coeffs)

    def impedance(self, omega):
        omega = np.atleast_2d(np.asarray(omega, dtype=float))  # (Nf, n)
        Nf, n = omega.shape

        # build the inverse-inductance matrix (1/H)
        Linv = self.coeffs.get(0, 0.0) * np.eye(n, dtype=complex)
        for p, c in self.coeffs.items():
            if p == 0:
                continue
            Linv = Linv + c * band_coupling_phased(n, p, self.theta)

        # invert to get L, then Z = j*Omega*L
        Lmat = np.linalg.inv(Linv)                      # (n, n)
        return 1j * omega[:, :, None] * Lmat[None, :, :]


class _Combination(Component):
    """Backing implementation for Component.series / Component.parallel."""

    def __init__(self, components: tuple[Component, ...], mode: str):
        self.components = components
        self.mode = mode

    def impedance(self, omega):
        if self.mode == "series":
            return _sum_immittances([c.impedance(omega) for c in self.components])
        return super().impedance(omega)

    def admittance(self, omega):
        if self.mode == "parallel":
            return _sum_immittances([c.admittance(omega) for c in self.components])
        return super().admittance(omega)


def _kron_list(mats):
    """Kronecker product of a list of matrices, left to right."""
    return reduce(np.kron, mats)


class ModulatedInductorMultiPump(Component):
    """
    Josephson inductor modulated by P independent pump tones.

    Each pump j couples its own sideband index k_j through a band-coupling
    matrix in the j-th tensor slot. The full sideband space is the tensor
    product of the per-pump sideband ladders, of dimension

        D = prod_j n_j,   n_j = 2*Kmax_j + 1

    The inductance to first order in each modulation depth is

        L / L0 = I_D
               + sum_j  eps_j * ( I_1 (x) ... (x) B1_j (x) ... (x) I_P )
               + (higher single-pump harmonics if `order` > 1)

    Only single-pump hops appear here. A single interaction
    is one hop in one index -- the modulation is a *sum* of cosines, never a
    product -- so there is NO eps_i*eps_j "diagonal" cross term inserted by
    hand. Two-index transitions (combination idlers omega_s + sum_j k_j w_pj)
    emerge from cascading many cells, i.e. from products of these single-hop
    operators, not from this per-cell matrix.

    Parameters
    ----------
    L0 : float
        Bare inductance L_J.
    eps : sequence of P floats
        Modulation depth per pump.
    n_sidebands : sequence of P ints
        Per-pump ladder size n_j = 2*Kmax_j + 1 (must be odd if the ladder is
        centred on the signal; any n_j works for the matrix machinery).
    order : int
        Highest single-pump harmonic to include (1 -> B1 only, 2 -> +B2, ...).
    coeffs : dict or sequence of dicts, optional
        Expansion coefficients {p: a_p}. A single dict applies to every pump;
        a list gives per-pump coefficients. Defaults to {p: 1.0}.
    theta : sequence of P floats, optional
        Per-pump propagation phase at this cell's position (e.g.
        omega_pj/v_pj * z), same role as `theta` in ModulatedInductor.
        Defaults to 0.0 for every pump.
    """

    def __init__(self, L0, eps, n_sidebands, order=1, coeffs=None, theta=None):
        self.L0 = L0
        self.eps = list(eps)
        self.n_sidebands = list(n_sidebands)
        self.P = len(self.eps)
        if len(self.n_sidebands) != self.P:
            raise ValueError("eps and n_sidebands must have the same length P")
        self.theta = list(theta) if theta is not None else [0.0] * self.P
        if len(self.theta) != self.P:
            raise ValueError("theta must have the same length P")

        # normalise coeffs -> one dict per pump
        if coeffs is None:
            base = {p: 1.0 for p in range(1, order + 1)}
            self.coeffs = [dict(base) for _ in range(self.P)]
        elif isinstance(coeffs, dict):
            self.coeffs = [dict(coeffs) for _ in range(self.P)]
        else:
            if len(coeffs) != self.P:
                raise ValueError("per-pump coeffs list must have length P")
            self.coeffs = [dict(c) for c in coeffs]

        self.order = order
        self.D = int(np.prod(self.n_sidebands))

    def _build_L_over_L0(self) -> np.ndarray:
        """Assemble the (D, D) coupling matrix  L / L0."""
        dims = self.n_sidebands
        Is = [np.eye(d, dtype=complex) for d in dims]

        L = np.eye(self.D, dtype=complex)  # identity term

        for j in range(self.P):
            for p, a in self.coeffs[j].items():
                mats = list(Is)  # identities in every slot
                mats[j] = band_coupling_phased(
                    dims[j], p, self.theta[j]
                )  # hop in slot j
                L = L + (self.eps[j] ** p) * a * _kron_list(mats)

        return L

    def impedance(self, omega):
        """
        omega : per-sideband frequency grid on the FULL tensor lattice.
            shape (D,) for one signal frequency, or (Nf, D) batched over Nf.
            Entry at flat index m must be  omega_s + sum_j k_j * omega_pj  for
            the (k_1,...,k_P) tuple that flat index m corresponds to (kron order,
            first pump = slowest-varying index).
        """
        omega = np.atleast_2d(np.asarray(omega, dtype=float))  # (Nf, D)
        Nf, D = omega.shape
        if D != self.D:
            raise ValueError(
                f"omega trailing dim {D} != tensor sideband dim {self.D} "
                f"(n_sidebands={self.n_sidebands})"
            )

        L = self._build_L_over_L0()  # (D, D)
        # Omega = diag(omega); Omega @ M is row-scaling, not a real matmul --
        # done via broadcasting instead of materializing a mostly-zero
        # (Nf,D,D) diagonal matrix and paying an O(Nf D^3) batched matmul
        # for what's really an O(Nf D^2) elementwise scale. D grows
        # multiplicatively with pump count, so this is the dominant cost
        # for multi-pump cells.
        #
        # Always keep the batch axis, even when Nf == 1 -- see the matching
        # comment in ModulatedInductor.impedance.
        return 1j * omega[:, :, None] * (self.L0 * L)[None, :, :]  # (Nf, D, D)


def n_sidebands_from_Kmax(Kmax) -> list[int]:
    """Per-pump ladder size n_j = k_max - k_min + 1, from a Kmax list of
    (k_min, k_max) pairs."""
    return [k_max - k_min + 1 for k_min, k_max in Kmax]


def multipump_frequency_grid(omega_s, omega_p, Kmax):
    """
    Build the per-sideband frequency vector on the tensor lattice, in the
    same kron flattening order the coupling matrix uses (first pump = outer).

    Kmax : sequence of P (k_min, k_max) pairs, e.g. [(-2, 3), (-1, 1)] --
        pump j tracks sidebands k_min_j..k_max_j (need not be symmetric
        about 0). ModulatedInductorMultiPump's coupling matrix hops by
        ladder *position*, so each pump's range must stay contiguous
        unit-spaced integers to match a physical sideband shift.

    Returns
    -------
    omega_grid : (D,) array of  omega_s + sum_j k_j * omega_pj
    labels     : list of (k_1,...,k_P) tuples in matching order
    """
    from itertools import product

    ranges = [range(k_min, k_max + 1) for k_min, k_max in Kmax]
    labels = list(product(*ranges))
    omega_p = np.asarray(omega_p, dtype=float)
    omega_grid = np.array(
        [omega_s + sum(k * w for k, w in zip(tup, omega_p)) for tup in labels],
        dtype=float,
    )
    return omega_grid, labels


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=160)

    # two pumps, Kmax = 1 each -> n_j = 3, D = 9
    omega_s = 6.0
    omega_p = [12.0, 7.0]
    eps = [0.05, 0.03]
    Kmax = [1, 1]
    n_sb = [2 * K + 1 for K in Kmax]

    comp = ModulatedInductorMultiPump(L0=1.0, eps=eps, n_sidebands=n_sb, order=1)
    Lrel = comp._build_L_over_L0()
    print("L / L0  (single hops only, no eps1*eps2 cross term):")
    print(Lrel.real)

    grid, labels = multipump_frequency_grid(omega_s, omega_p, Kmax)
    print("\n(k1,k2) -> omega:")
    for lab, w in zip(labels, grid):
        print(f"   {lab} -> {w:6.2f}")

    Z = comp.impedance(grid)
    print("\nimpedance shape:", Z.shape)

    # confirm: (0,0)->(1,1) entry is ZERO (no single-cell double hop)
    lut = {lab: i for i, lab in enumerate(labels)}
    print(
        "\n(0,0)->(1,0) coupling in L/L0 =",
        Lrel[lut[(0, 0)], lut[(1, 0)]].real,
        "(eps1)",
    )
    print(
        "(0,0)->(0,1) coupling in L/L0 =", Lrel[lut[(0, 0)], lut[(0, 1)]].real, "(eps2)"
    )
    print(
        "(0,0)->(1,1) coupling in L/L0 =",
        Lrel[lut[(0, 0)], lut[(1, 1)]].real,
        "(must be 0: not a single hop)",
    )
