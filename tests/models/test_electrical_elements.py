import numpy as np
import pytest

from models.electrical_elements import (
    Capacitor,
    Component,
    Inductor,
    ModulatedInductor,
    band_coupling_phased,
    multipump_frequency_grid,
    n_sidebands_from_Kmax,
)


class TestBandCouplingPhased:
    def test_offset_zero_is_all_zero(self):
        M = band_coupling_phased(4, 0, theta=0.7)
        np.testing.assert_array_equal(M, np.zeros((4, 4)))

    def test_offset_at_or_beyond_n_is_all_zero(self):
        for offset in (4, 5, -4):
            M = band_coupling_phased(4, offset, theta=0.3)
            np.testing.assert_array_equal(M, np.zeros((4, 4)))

    def test_hermitian(self):
        """Up-shift/down-shift entries must be a conjugate pair, or the
        coupling matrix isn't passive/lossless."""
        M = band_coupling_phased(5, 2, theta=1.234)
        np.testing.assert_allclose(M, M.conj().T, atol=1e-14)

    def test_matches_closed_form(self):
        n, offset, theta = 5, 2, 0.9
        M = band_coupling_phased(n, offset, theta)
        expected = np.zeros((n, n), dtype=complex)
        for i in range(n - offset):
            expected[i, i + offset] = 0.5 * np.exp(1j * offset * theta)
            expected[i + offset, i] = 0.5 * np.exp(-1j * offset * theta)
        np.testing.assert_allclose(M, expected, atol=1e-14)

    def test_negative_offset_matches_positive(self):
        M_pos = band_coupling_phased(5, 2, theta=0.4)
        M_neg = band_coupling_phased(5, -2, theta=0.4)
        np.testing.assert_array_equal(M_pos, M_neg)


class TestBasicComponents:
    def test_inductor_impedance(self):
        L = 2.5
        omega = np.array([1.0, 2.0, 3.0])
        Z = Inductor(L).impedance(omega)
        np.testing.assert_allclose(Z, 1j * omega * L)

    def test_inductor_admittance_is_reciprocal(self):
        L = 1.7
        omega = np.array([1.0, 5.0])
        ind = Inductor(L)
        np.testing.assert_allclose(ind.admittance(omega), 1.0 / ind.impedance(omega))

    def test_capacitor_impedance(self):
        C = 3.0
        omega = np.array([1.0, 2.0])
        Z = Capacitor(C).impedance(omega)
        np.testing.assert_allclose(Z, 1.0 / (1j * omega * C))

    def test_impedance_matrix_promotes_unmodulated_component_to_diagonal(self):
        """A plain (non-pump-coupled) component doesn't mix sidebands, so
        its harmonic matrix must be exactly diag(Z(omega_1), Z(omega_2), ...)."""
        omega = np.array([[1.0, 2.0, 3.0]])  # (Nf=1, n=3) distinct sideband freqs
        L = Inductor(2.0)
        Zmat = L.impedance_matrix(omega)
        expected = np.zeros((1, 3, 3), dtype=complex)
        expected[0] = np.diag(L.impedance(omega)[0])
        np.testing.assert_allclose(Zmat, expected, atol=1e-14)


class TestComponentCombinations:
    def test_series_sums_impedance(self):
        omega = np.array([1.0, 2.0])
        L, C = Inductor(2.0), Capacitor(0.5)
        combo = Component.series(L, C)
        np.testing.assert_allclose(
            combo.impedance(omega), L.impedance(omega) + C.impedance(omega)
        )

    def test_parallel_sums_admittance(self):
        omega = np.array([1.0, 2.0])
        L, C = Inductor(2.0), Capacitor(0.5)
        combo = Component.parallel(L, C)
        np.testing.assert_allclose(
            combo.admittance(omega), L.admittance(omega) + C.admittance(omega)
        )

    def test_series_impedance_via_admittance_inverse(self):
        """series admittance falls back to 1/impedance (Component.admittance)."""
        omega = np.array([1.0, 3.0])
        combo = Component.series(Inductor(1.0), Inductor(1.0))
        np.testing.assert_allclose(combo.admittance(omega), 1.0 / combo.impedance(omega))

    def test_parallel_impedance_via_admittance_inverse(self):
        omega = np.array([1.0, 3.0])
        combo = Component.parallel(Capacitor(1.0), Capacitor(2.0))
        np.testing.assert_allclose(combo.impedance(omega), 1.0 / combo.admittance(omega))


class TestModulatedInductor:
    def test_coeffs_dc_only_matches_plain_inductor(self):
        """A pure DC term (no harmonics) is just a diagonal Inductor(1/c0)."""
        L0 = 4.0
        omega = np.full((1, 3), 2.0)  # (Nf=1, n=3)
        mi = ModulatedInductor(coeffs={0: 1.0 / L0}, theta=0.0)
        Z = mi.impedance(omega)
        expected = 1j * omega[:, :, None] * (L0 * np.eye(3))[None, :, :]
        np.testing.assert_allclose(Z, expected)

    def test_eps_zero_order_matches_plain_inductor(self):
        L0 = 3.0
        omega = np.full((1, 2), 5.0)
        mi = ModulatedInductor(eps=0.0, order=1, L0=L0, theta=1.1)
        Z = mi.impedance(omega)
        expected = 1j * omega[:, :, None] * (L0 * np.eye(2))[None, :, :]
        np.testing.assert_allclose(Z, expected)

    def test_coeffs_harmonics_match_closed_form_inverse(self):
        """Z must equal j*Omega * inv(c0*I + c1*band_coupling_phased(n,1,theta))."""
        c0, c1, theta = 1.0, 0.3, 0.5
        n = 3
        mi = ModulatedInductor(coeffs={0: c0, 1: c1}, theta=theta, order=1)
        omega = np.full((1, n), 2.0)

        Linv = c0 * np.eye(n) + c1 * band_coupling_phased(n, 1, theta)
        expected = 1j * omega[0, 0] * np.linalg.inv(Linv)

        Z = mi.impedance(omega)
        np.testing.assert_allclose(Z[0], expected, atol=1e-12)

    def test_eps_first_order_matches_closed_form(self):
        """L/L0 = I + eps*band_coupling_phased(n,1,theta); Z = j*Omega*L0*L/L0."""
        eps, L0, theta = 0.2, 1.5, 0.5
        n = 3
        mi = ModulatedInductor(eps=eps, order=1, L0=L0, theta=theta)
        omega = np.full((1, n), 2.0)

        L_over_L0 = np.eye(n) + eps * band_coupling_phased(n, 1, theta)
        expected = 1j * omega[0, 0] * L0 * L_over_L0

        Z = mi.impedance(omega)
        np.testing.assert_allclose(Z[0], expected, atol=1e-12)


class TestMultipumpFrequencyGrid:
    def test_labels_and_grid_values(self):
        omega_s = 6.0
        omega_p = [12.0, 7.0]
        Kmax = [(-1, 1), (0, 1)]
        grid, labels = multipump_frequency_grid(omega_s, omega_p, Kmax)

        expected_labels = [(-1, 0), (-1, 1), (0, 0), (0, 1), (1, 0), (1, 1)]
        assert labels == expected_labels

        expected_grid = np.array(
            [omega_s + k1 * 12.0 + k2 * 7.0 for k1, k2 in expected_labels]
        )
        np.testing.assert_allclose(grid, expected_grid)

    def test_single_pump_reduces_to_flat_ladder(self):
        grid, labels = multipump_frequency_grid(2.0, [3.0], [(-1, 1)])
        np.testing.assert_allclose(grid, [-1.0, 2.0, 5.0])
        assert labels == [(-1,), (0,), (1,)]


class TestNSidebandsFromKmax:
    def test_basic(self):
        assert n_sidebands_from_Kmax([(-2, 3), (-1, 1)]) == [6, 3]

    def test_single_pump(self):
        assert n_sidebands_from_Kmax([(0, 0)]) == [1]
