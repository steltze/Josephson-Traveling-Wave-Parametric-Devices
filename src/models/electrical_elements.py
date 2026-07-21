from __future__ import annotations

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


def _is_matrix(value: np.ndarray) -> bool:
    """True for a genuine (..., n, n) sideband-coupling matrix, as opposed
    to a per-sideband array of independent scalar values — which, for a
    batched (Nf, n) grid, also has ndim 2 but isn't square in general."""
    return value.ndim >= 2 and value.shape[-1] == value.shape[-2]


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
    Pump-modulated (Josephson) inductor.

    The bare inductance is modulated by a pump tone, coupling sidebands
    through the (n, n) matrix

        L / L0 = I + sum_{p=1}^{order} eps^p * coeffs[p] * band_coupling(n, p)

    impedance(omega) = j * Omega @ (L0 * L/L0), where Omega = diag(omega)
    over the trailing axis of `omega`. Unlike a plain Inductor/Capacitor,
    `omega` here must already be the per-sideband frequency grid — shape
    (n,) for one signal frequency or (Nf, n) batched — since building that
    grid (e.g. omega_signal + k*omega_pump for k in ks_state) requires
    knowing which sidebands the surrounding circuit tracks, not just this
    one element.
    """

    def __init__(self, L0, eps, order=1, coeffs=None):
        self.L0 = L0
        self.eps = eps
        self.coeffs = coeffs if coeffs is not None else {p: 1.0 for p in range(1, order + 1)}

    def impedance(self, omega):
        omega = np.atleast_2d(np.asarray(omega, dtype=float))  # (Nf, n)
        Nf, n = omega.shape

        Omega = np.zeros((Nf, n, n), dtype=complex)
        idx = np.arange(n)
        Omega[:, idx, idx] = omega

        L = np.eye(n, dtype=complex)
        for p, a in self.coeffs.items():
            L = L + (self.eps ** p) * a * band_coupling(n, p)

        Z = 1j * Omega @ (self.L0 * L)
        return Z[0] if Z.shape[0] == 1 else Z


class InSeries(Component):
    def __init__(self, *components: Component):
        self.components = components

    def impedance(self, omega):
        return _sum_immittances([c.impedance(omega) for c in self.components])


class Parallel(Component):
    def __init__(self, *components: Component):
        self.components = components

    def admittance(self, omega):
        return _sum_immittances([c.admittance(omega) for c in self.components])
