from __future__ import annotations

from functools import reduce

import numpy as np


def band_coupling_phased(n: int, offset: int, theta) -> np.ndarray:
    """(..., n, n) complex band-coupling matrix(es) for a pump modulation
    cos(w_p t - theta). `theta` is a scalar (returns (n, n)) or an array,
    e.g. one theta per cell (returns (*theta.shape, n, n)) -- batches over
    whatever leading shape `theta` carries instead of building one matrix
    per call, so e.g. JTLDiscrete.build can construct every cell's coupling
    matrix in one vectorized call.

    Entry [..., target, source]: a target sideband above its source
    (up-shift, towards +w_p) picks up exp(-i*offset*theta); a target below
    its source (down-shift) picks up the conjugate exp(+i*offset*theta).
    Must stay a conjugate pair (not the same phase on both diagonals) or the
    sideband-coupling matrix stops being passive/lossless (fails Manley-Rowe
    / pseudo-unitarity).
    """
    theta = np.asarray(theta, dtype=float)
    out = np.zeros(theta.shape + (n, n), dtype=complex)
    if abs(offset) >= n or offset == 0:
        return out
    idx = np.arange(n - abs(offset))
    phase = np.exp(1j * abs(offset) * theta)  # theta.shape
    out[..., idx, idx + abs(offset)] = 0.5 * phase[..., None]
    out[..., idx + abs(offset), idx] = 0.5 * np.conj(phase)[..., None]
    return out


def _trail(x) -> np.ndarray:
    """x, shape (*batch,) -> (*batch, 1, 1) -- pads two trailing singleton
    axes so a per-cell (or unbatched, scalar) coefficient broadcasts against
    a (*batch, n, n) matrix."""
    x = np.asarray(x)
    return x.reshape(x.shape + (1, 1))


def _trail_like(x, omega: np.ndarray) -> np.ndarray:
    """x, shape (*batch,) -> (*batch, 1, ..., 1) with one trailing singleton
    per axis of `omega` -- pads a per-cell (or unbatched, scalar) component
    value so it broadcasts against `omega`, whatever `omega`'s shape is
    (Inductor/Capacitor, unlike ModulatedInductor, don't force omega to a
    particular ndim -- see their docstrings)."""
    x = np.asarray(x)
    omega = np.asarray(omega)
    return x.reshape(x.shape + (1,) * omega.ndim)


def _impedance_from_inductance(Lmat: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Z[..., f, a, b] = j * omega[f, a] * Lmat[..., a, b].

    Lmat: (*batch, n, n); omega: (Nf, n). Returns (*batch, Nf, n, n) --
    generalizes `Z = j*Omega*L` (each sideband row scaled by its own omega)
    to an arbitrary leading batch shape on `Lmat` (e.g. one L per cell), not
    just a single (n, n) matrix.
    """
    Nf, n = omega.shape
    omega_b = omega.reshape((1,) * (Lmat.ndim - 2) + (Nf, n, 1))
    Lmat_b = Lmat[..., None, :, :]
    return 1j * omega_b * Lmat_b


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


def _sum_immittances(pairs: list[tuple[np.ndarray, bool]]) -> np.ndarray:
    """
    Whether a value needs diagonal-promotion before summing is looked up
    from each component's own static `couples_sidebands` flag (passed in by
    the caller), not guessed from the value's array shape: shape alone can't
    reliably tell a diagonal (..., Nf, n) representation apart from a
    genuine (..., n, n) matrix whenever Nf happens to equal n, and that
    ambiguity gets worse, not better, once a per-cell batch axis is added on
    top (see JTLDiscrete.build) -- so this never inspects `value.shape` to
    decide, only `couples_sidebands`.
    """
    values = [np.asarray(v) for v, _ in pairs]
    matrices = [v for v, is_matrix in pairs if is_matrix]
    if not matrices:
        return sum(values)

    ns = {m.shape[-1] for m in matrices}
    if len(ns) > 1:
        raise ValueError(f"sideband dimensions don't match across components: {ns}")

    total = np.zeros(matrices[0].shape, dtype=complex)
    for v, is_matrix in pairs:
        total = total + (v if is_matrix else _diag_promote(v))
    return total


class Component:
    """Base class for two-terminal circuit elements.

    A value is a float/complex (or an array over frequency) for plain
    elements, or an (n, n) sideband-coupling matrix (or (Nf, n, n) when
    batched over frequency) for elements modulated by a pump tone.
    impedance/admittance are duals of each other via matrix/scalar inversion.
    """

    couples_sidebands = False

    def impedance(self, omega):
        Y = self.admittance(omega)
        return np.linalg.inv(Y) if self.couples_sidebands else 1.0 / Y

    def admittance(self, omega):
        Z = self.impedance(omega)
        return np.linalg.inv(Z) if self.couples_sidebands else 1.0 / Z

    def impedance_matrix(self, omega):
        """impedance(omega) as an (..., n, n) matrix, diagonal if this
        component doesn't couple sidebands."""
        Z = np.asarray(self.impedance(omega))
        return Z if self.couples_sidebands else _diag_promote(Z)

    def admittance_matrix(self, omega):
        """admittance(omega) as an (..., n, n) matrix, diagonal if this
        component doesn't couple sidebands."""
        Y = np.asarray(self.admittance(omega))
        return Y if self.couples_sidebands else _diag_promote(Y)

    @staticmethod
    def series(*components: Component) -> Component:
        return _Combination(components, "series")

    @staticmethod
    def parallel(*components: Component) -> Component:
        return _Combination(components, "parallel")


class Inductor(Component):
    def __init__(self, L: float):
        self.L = L

    def impedance(self, omega):
        """
        `L` may be a scalar (single cell) or an array, e.g. one per cell, in
        which case the result batches over that leading shape: (*batch,
        *omega.shape) -- `_trail_like` appends one trailing singleton axis
        per axis of `omega` so a per-cell L broadcasts against it, whatever
        `omega`'s shape is (an unbatched, e.g. scalar, L broadcasts fine
        either way, so this doesn't change the single-cell result).
        """
        omega = np.asarray(omega)
        return 1j * omega * _trail_like(self.L, omega)


class Capacitor(Component):
    def __init__(self, C: float):
        self.C = C

    def impedance(self, omega):
        """`C` may be a scalar or an array (e.g. one per cell) -- see
        Inductor.impedance for the batching/broadcasting convention."""
        omega = np.asarray(omega)
        return 1.0 / (1j * omega * _trail_like(self.C, omega))


class ModulatedInductor(Component):
    """
    Pump-modulated (Josephson) inductor. Two calling conventions, chosen by
    whether `eps` is given:

    - `coeffs` (exact): `coeffs` are the ABSOLUTE inverse-inductance
      harmonics, in 1/H --

          Linv(t) = sum_{p>=0} c[p] * cos(p*theta)

      so the inverse-inductance matrix over sidebands is

          Linv_mat = c[0]*I + sum_{p>=1} c[p] * band_coupling_phased(n, p, theta)

      and the series impedance is  Z = j*Omega * inv(Linv_mat). `coeffs`
      must include the DC term as coeffs[0]. This is what a Jacobi-Anger/
      Bessel expansion produces directly -- see JTLDiscrete/
      JTLDiscreteSQUID.build.

    - `eps` (perturbative, the original form this class had before the
      `coeffs` convention above): a small modulation-depth expansion of the
      inductance itself (not its inverse) --

          L(t)/L0 = I + sum_p eps^p * coeffs[p] * band_coupling_phased(n, p, theta)

      and Z = j*Omega * (L0 * L/L0). Here `coeffs` (if given) are RELATIVE
      per-harmonic weights, defaulting to 1.0 for p=1..order -- not absolute
      1/H values as in the coeffs-only path above. `theta` is the pump's
      propagation phase at this cell's position (e.g. omega_pump/v_pump * z),
      applied via band_coupling_phased just like the coeffs-only path.
    """

    couples_sidebands = True

    def __init__(self, coeffs=None, theta=0.0, order=None, L0=None, eps=None):
        self.theta = theta
        self.L0 = L0
        self.eps = eps
        if eps is not None:
            self.order = order if order is not None else 1
            self.coeffs = coeffs if coeffs is not None else {p: 1.0 for p in range(1, self.order + 1)}
        else:
            self.coeffs = coeffs           # {0: c0, 1: c1, ...} in 1/H
            self.order = order if order is not None else max(coeffs)

    def impedance(self, omega):
        """
        `theta`/`eps`/`L0`/`coeffs` values may each be a scalar (single
        cell -- the original contract) or an array sharing a common leading
        batch shape (e.g. one value per cell), in which case the result
        batches over that shape: (*batch, Nf, n, n) instead of (Nf, n, n).
        This lets a caller build every cell's impedance in one call instead
        of constructing one ModulatedInductor per cell.
        """
        omega = np.atleast_2d(np.asarray(omega, dtype=float))  # (Nf, n)
        Nf, n = omega.shape
        theta = np.asarray(self.theta, dtype=float)

        if self.eps is not None:
            eps = np.asarray(self.eps, dtype=float)
            L = np.eye(n, dtype=complex)
            for p, a in self.coeffs.items():
                Bp = band_coupling_phased(n, p, theta)  # theta.shape + (n, n)
                L = L + _trail(eps**p) * _trail(a) * Bp
            Lmat = _trail(self.L0) * L                  # (*batch, n, n)
            return _impedance_from_inductance(Lmat, omega)

        # build the inverse-inductance matrix (1/H)
        Linv = self.coeffs.get(0, 0.0) * np.eye(n, dtype=complex)
        for p, c in self.coeffs.items():
            if p == 0:
                continue
            Bp = band_coupling_phased(n, p, theta)  # theta.shape + (n, n)
            Linv = Linv + _trail(c) * Bp

        # invert to get L, then Z = j*Omega*L -- np.linalg.inv batches over
        # any leading dims, so this is one call regardless of batch shape.
        Lmat = np.linalg.inv(Linv)
        return _impedance_from_inductance(Lmat, omega)


class _Combination(Component):
    """Backing implementation for Component.series / Component.parallel."""

    def __init__(self, components: tuple[Component, ...], mode: str):
        self.components = components
        self.mode = mode

    @property
    def couples_sidebands(self) -> bool:
        return any(c.couples_sidebands for c in self.components)

    def impedance(self, omega):
        if self.mode == "series":
            return _sum_immittances(
                [(c.impedance(omega), c.couples_sidebands) for c in self.components]
            )
        return super().impedance(omega)

    def admittance(self, omega):
        if self.mode == "parallel":
            return _sum_immittances(
                [(c.admittance(omega), c.couples_sidebands) for c in self.components]
            )
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

    couples_sidebands = True

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
