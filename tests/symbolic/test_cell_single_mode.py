import numpy as np
import pytest
from sympy import exp, I, simplify, expand, Function

from src.symbolic.cell_single_mode import CellSingleMode
from src.models.cell import CellImmitance


def _setup_cell(cls):
    c = CellSingleMode()
    cls.fourier_basis = c._fourier_basis
    cls.make_harmonic_functions = c._make_harmonic_functions
    cls.build_Zs_series = c._build_Zs_series
    cls.build_Yg_shunt = c._build_Yg_shunt
    cls.extract_harmonic_results = c._extract_harmonic_results
    cls.build_symbolic_transfer_matrix = c.build_symbolic_transfer_matrix
    cls.build_numeric_matrix = c.build_numeric_matrix
    cls.t = c.t
    cls.theta = c.theta
    cls.omega_p = c.omega_p
    cls.k = c.k
    cls.n = c.n
    cls.omega = c.omega
    cls.Zs0 = c.Zs0
    cls.Yg0 = c.Yg0
    cls.V = c.V
    cls.Ic = c.Ic
    cls.xi = c.xi


class TestFourierBasis:
    @classmethod
    def setup_class(cls):
        _setup_cell(cls)

    def test_zero_harmonic_is_unity(self):
        assert self.fourier_basis(0) == 1

    def test_positive_harmonic(self):
        assert self.fourier_basis(1) == exp(I * self.omega_p * self.t)

    def test_conjugate_pair_product_is_unity(self):
        product = expand(self.fourier_basis(2) * self.fourier_basis(-2))
        assert simplify(product) == 1

    def test_additivity_of_exponents(self):
        assert (
            simplify(
                self.fourier_basis(1) * self.fourier_basis(2) - self.fourier_basis(3)
            )
            == 0
        )

    def test_negative_is_conjugate_of_positive(self):
        diff = simplify(self.fourier_basis(-1) - exp(-I * self.omega_p * self.t))
        assert diff == 0


class TestSeriesStructure:
    @classmethod
    def setup_class(cls):
        _setup_cell(cls)

    def test_Zs_series_M0_is_dc_only(self):
        """With no harmonic functions (M=0), Zs(t) reduces to the DC term Zs0(xi)."""
        assert self.build_Zs_series([]) == self.Zs0(self.xi)

    def test_Yg_series_M0_is_dc_only(self):
        assert self.build_Yg_shunt([]) == self.Yg0(self.xi)

    def test_Zs_series_M1_contains_Zs1(self):
        Zs_m, _ = self.make_harmonic_functions(1)
        Zs_t = self.build_Zs_series(Zs_m)
        Zs1 = Function("Zs1")
        assert Zs_t.has(Zs1(self.xi))

    def test_Yg_series_M1_contains_Yg1(self):
        _, Yg_m = self.make_harmonic_functions(1)
        Yg_t = self.build_Yg_shunt(Yg_m)
        Yg1 = Function("Yg1")
        assert Yg_t.has(Yg1(self.xi))

    def test_Zs_series_M1_has_three_terms(self):
        """DC + E(+1) term + E(-1) term -> 3 additive pieces after expansion."""
        Zs_m, _ = self.make_harmonic_functions(1)
        Zs_t = expand(self.build_Zs_series(Zs_m))
        assert len(Zs_t.as_ordered_terms()) == 3

    def test_Zs_series_M1_DC_coefficient_is_Zs0(self):
        """The coefficient of E(0)=1 inside Zs(t) is just Zs0(xi)."""
        Zs_m, _ = self.make_harmonic_functions(1)
        Zs_t = expand(self.build_Zs_series(Zs_m))
        dc_part = (
            Zs_t
            - Zs_t.coeff(self.fourier_basis(1)) * self.fourier_basis(1)
            - Zs_t.coeff(self.fourier_basis(-1)) * self.fourier_basis(-1)
        )
        assert simplify(dc_part - self.Zs0(self.xi)) == 0


class TestHarmonicExtraction:
    @classmethod
    def setup_class(cls):
        _setup_cell(cls)

    @pytest.fixture(scope="class")
    def results_M0(self):
        return self.extract_harmonic_results(
            0, self.build_Zs_series([]), self.build_Yg_shunt([])
        )

    @pytest.fixture(scope="class")
    def results_M1(self):
        Zs_m, Yg_m = self.make_harmonic_functions(1)
        return self.extract_harmonic_results(
            1, self.build_Zs_series(Zs_m), self.build_Yg_shunt(Yg_m)
        )

    def test_M0_only_dc_key(self, results_M0):
        """At M=0 (no pump harmonics) only the DC key j=0 should be present."""
        assert set(results_M0.keys()) == {0}

    def test_M1_has_keys_minus2_to_plus2(self, results_M1):
        assert set(results_M1.keys()) == {-2, -1, 0, 1, 2}

    def test_M0_dc_voltage(self, results_M0):
        """V[k,n+1] DC = V[k,n] - Zs0*I[k,n]."""
        cv0, _ = results_M0[0]
        expected = (
            self.V[self.k, self.n]
            - self.Zs0(self.omega[self.k]) * self.Ic[self.k, self.n]
        )
        assert simplify(expand(cv0 - expected)) == 0

    def test_M0_dc_current(self, results_M0):
        """I[k,n+1] DC = -Yg0*V[k,n] + (1 + Yg0*Zs0)*I[k,n]."""
        _, ci0 = results_M0[0]
        Zs0k = self.Zs0(self.omega[self.k])
        Yg0k = self.Yg0(self.omega[self.k])
        expected = (
            -Yg0k * self.V[self.k, self.n] + (1 + Yg0k * Zs0k) * self.Ic[self.k, self.n]
        )
        assert simplify(expand(ci0 - expected)) == 0

    def test_M1_j1_voltage_coeff_involves_Zs1(self, results_M1):
        """The j=+1 voltage harmonic must contain a Zs1 function."""
        cv1, _ = results_M1[1]
        Zs1 = Function("Zs1")
        assert cv1.has(Zs1)

    def test_M1_dc_has_no_exponentials(self, results_M1):
        """The DC result must not contain any Floquet exponentials."""
        cv0, ci0 = results_M1[0]
        for j in [-2, -1, 1, 2]:
            assert cv0.coeff(self.fourier_basis(j)) == 0
            assert ci0.coeff(self.fourier_basis(j)) == 0


class TestSymbolicTransferMatrix:
    @classmethod
    def setup_class(cls):
        _setup_cell(cls)

    @pytest.fixture(scope="class")
    def T_M0_N0(self):
        """2x2 transfer matrix for M=0 (no pump), single mode k=0."""
        T_sym, state_syms, _, __ = self.build_symbolic_transfer_matrix(
            M=0, ks_state=[0]
        )
        return T_sym, state_syms

    @pytest.fixture(scope="class")
    def T_M1_N1(self):
        """4x4 transfer matrix for M=1, sidebands [0,1]."""
        T_sym, state_syms, _, __ = self.build_symbolic_transfer_matrix(
            M=1, ks_state=[0, 1]
        )
        return T_sym, state_syms

    def test_shape_M0_N0(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert T_sym.shape == (2, 2)

    def test_shape_M1_N1(self, T_M1_N1):
        T_sym, _ = T_M1_N1
        assert T_sym.shape == (4, 4)

    def test_M0_T00_is_one(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert T_sym[0, 0] == 1

    def test_M0_T01_is_minus_Zs0(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert simplify(T_sym[0, 1] + self.Zs0(self.omega[0])) == 0

    def test_M0_T10_is_minus_Yg0(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert simplify(T_sym[1, 0] + self.Yg0(self.omega[0])) == 0

    def test_M0_T11_is_one_plus_Yg0_Zs0(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        Zs0k = self.Zs0(self.omega[0])
        Yg0k = self.Yg0(self.omega[0])
        assert simplify(T_sym[1, 1] - (1 + Yg0k * Zs0k)) == 0

    def test_M0_determinant_is_one(self, T_M0_N0):
        """Reciprocal network: det(T) = AD - BC = 1."""
        T_sym, _ = T_M0_N0
        det = T_sym[0, 0] * T_sym[1, 1] - T_sym[0, 1] * T_sym[1, 0]
        assert simplify(expand(det) - 1) == 0

    def test_M1_diagonal_voltage_rows_are_one(self, T_M1_N1):
        """T[0,0] and T[1,1] are both 1 (V[j,n] -> V[j,n+1] direct coupling)."""
        T_sym, _ = T_M1_N1
        assert T_sym[0, 0] == 1
        assert T_sym[1, 1] == 1

    def test_M1_voltage_rows_have_no_V_cross_coupling(self, T_M1_N1):
        """V[0,n+1] does not depend on V[1,n] (T[0,1] = 0)."""
        T_sym, _ = T_M1_N1
        assert T_sym[0, 1] == 0

    def test_M1_T02_is_minus_Zs0(self, T_M1_N1):
        """V[0,n+1] coefficient of I[0,n] is -Zs0(omega[0]) (T[0,2])."""
        T_sym, _ = T_M1_N1
        assert simplify(T_sym[0, 2] + self.Zs0(self.omega[0])) == 0

    def test_M1_T13_is_minus_Zs0(self, T_M1_N1):
        """Same -Zs0 appears in the second voltage row (T[1,3])."""
        T_sym, _ = T_M1_N1
        assert simplify(T_sym[1, 3] + self.Zs0(self.omega[1])) == 0

    def test_M1_current_row0_no_Zs2_terms(self, T_M1_N1):
        """With M=1, Zs2 should not appear anywhere in the matrix."""
        T_sym, _ = T_M1_N1
        Zs2 = Function("Zs2")
        assert not T_sym.has(Zs2)

    def test_state_syms_ordering(self, T_M1_N1):
        """State vector is grouped: [V[0,n], V[1,n], I[0,n], I[1,n]]."""
        _, state_syms = T_M1_N1
        assert state_syms[0] == self.V[0, self.n]
        assert state_syms[1] == self.V[1, self.n]
        assert state_syms[2] == self.Ic[0, self.n]
        assert state_syms[3] == self.Ic[1, self.n]

    # ------------------------------------------------------------------
    # M=2, ks=[0,1,2] — 6x6 matrix
    # ------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def T_M2_N2(self):
        """6x6 transfer matrix for M=2, sidebands [0,1,2]."""
        T_sym, state_syms, _, __ = self.build_symbolic_transfer_matrix(
            M=2, ks_state=[0, 1, 2]
        )
        return T_sym, state_syms

    def test_shape_M2_N2(self, T_M2_N2):
        T_sym, _ = T_M2_N2
        assert T_sym.shape == (6, 6)

    def test_M2_voltage_block_is_identity(self, T_M2_N2):
        """Upper-left 3x3 block: diagonal 1, no V-V cross coupling."""
        T_sym, _ = T_M2_N2
        for i in range(3):
            assert T_sym[i, i] == 1
            for j in range(3):
                if i != j:
                    assert T_sym[i, j] == 0

    def test_M2_T05_involves_Zs2(self, T_M2_N2):
        """V[0,n+1] → I[2,n] coupling (m=2 harmonic) must contain Zs2."""
        T_sym, _ = T_M2_N2
        assert T_sym[0, 5].has(Function("Zs2"))

    def test_M2_T23_involves_Zs2(self, T_M2_N2):
        """V[2,n+1] → I[0,n] coupling (m=2 harmonic) must contain Zs2."""
        T_sym, _ = T_M2_N2
        assert T_sym[2, 3].has(Function("Zs2"))

    def test_M2_no_Zs3(self, T_M2_N2):
        """With M=2, Zs3 must not appear anywhere (truncation)."""
        T_sym, _ = T_M2_N2
        assert not T_sym.has(Function("Zs3"))

    def test_M2_no_Yg3(self, T_M2_N2):
        """With M=2, Yg3 must not appear anywhere (truncation)."""
        T_sym, _ = T_M2_N2
        assert not T_sym.has(Function("Yg3"))

    def test_M2_state_syms_ordering(self, T_M2_N2):
        """State: [V[0,n], V[1,n], V[2,n], I[0,n], I[1,n], I[2,n]]."""
        _, state_syms = T_M2_N2
        for i, k in enumerate([0, 1, 2]):
            assert state_syms[i] == self.V[k, self.n]
            assert state_syms[3 + i] == self.Ic[k, self.n]


_OMEGA_S = 1.0
_OMEGA_P = 0.1
_THETA = np.pi / 4
_ZS0 = 1.0 + 0.0j
_YG0 = 0.5 + 0.0j
_ZS1 = 0.1 + 0.05j
_YG1 = 0.05 + 0.02j
_ZS2 = 0.02 + 0.01j
_YG2 = 0.01 + 0.005j


def _omega_val(q):
    return _OMEGA_S + q * _OMEGA_P


def _Zs_num(m, omega):
    return {1: _ZS1, 2: _ZS2}.get(m, 0 + 0j)


def _Yg_num(m, omega):
    return {1: _YG1, 2: _YG2}.get(m, 0 + 0j)


def _make_cell(theta=_THETA):
    return CellImmitance(
        theta=theta,
        Zs0_fn=lambda omega: _ZS0,
        Yg0_fn=lambda omega: _YG0,
        Zs_harm_fn=_Zs_num,
        Yg_harm_fn=_Yg_num,
    )


@pytest.fixture(scope="module")
def T_num_M1_N1():
    cell_obj = CellSingleMode()
    M, ks_state = 1, [0, 1]
    T_sym, state_syms, Zs_m, Yg_m = cell_obj.build_symbolic_transfer_matrix(M, ks_state)
    dim = len(state_syms)
    return cell_obj.build_numeric_matrix(
        T_sym,
        dim,
        M,
        ks_state,
        Zs_m,
        Yg_m,
        _make_cell(),
        _omega_val,
        _OMEGA_P,
        k_val=0,
    )


@pytest.fixture(scope="module")
def T_num_M0_N0():
    cell_obj = CellSingleMode()
    M, ks_state = 0, [0]
    T_sym, state_syms, Zs_m, Yg_m = cell_obj.build_symbolic_transfer_matrix(M, ks_state)
    dim = len(state_syms)
    return cell_obj.build_numeric_matrix(
        T_sym,
        dim,
        M,
        ks_state,
        Zs_m,
        Yg_m,
        _make_cell(),
        _omega_val,
        _OMEGA_P,
        k_val=0,
    )


class TestNumericalMatrix:
    def test_shape(self, T_num_M1_N1):
        assert T_num_M1_N1.shape == (4, 4)

    def test_dtype_is_complex(self, T_num_M1_N1):
        assert np.issubdtype(T_num_M1_N1.dtype, np.complexfloating)

    def test_T00_real_is_one(self, T_num_M1_N1):
        assert T_num_M1_N1[0, 0].real == pytest.approx(1.0, abs=1e-10)

    def test_T01_is_zero(self, T_num_M1_N1):
        assert abs(T_num_M1_N1[0, 1]) == pytest.approx(0.0, abs=1e-10)

    def test_T02_real_is_minus_Zs0(self, T_num_M1_N1):
        assert T_num_M1_N1[0, 2].real == pytest.approx(-_ZS0.real, abs=1e-6)

    def test_T03_real(self, T_num_M1_N1):
        # -Zs1(omega[1])*exp(-I*theta) real part ≈ -0.106066
        assert T_num_M1_N1[0, 3].real == pytest.approx(-0.106066, abs=1e-4)

    def test_T03_imag(self, T_num_M1_N1):
        # -Zs1(omega[1])*exp(-I*theta) imag part ≈ +0.035355
        assert T_num_M1_N1[0, 3].imag == pytest.approx(+0.035355, abs=1e-4)

    def test_T11_real_is_one(self, T_num_M1_N1):
        assert T_num_M1_N1[1, 1].real == pytest.approx(1.0, abs=1e-10)

    def test_T12_real(self, T_num_M1_N1):
        # -Zs1(omega[0])*exp(I*theta) real part ≈ -0.035355
        assert T_num_M1_N1[1, 2].real == pytest.approx(-0.035355, abs=1e-4)

    def test_T12_imag(self, T_num_M1_N1):
        # -Zs1(omega[0])*exp(I*theta) imag part ≈ -0.106066
        assert T_num_M1_N1[1, 2].imag == pytest.approx(-0.106066, abs=1e-4)

    def test_T13_real_is_minus_Zs0(self, T_num_M1_N1):
        assert T_num_M1_N1[1, 3].real == pytest.approx(-_ZS0.real, abs=1e-6)

    def test_T20_real(self, T_num_M1_N1):
        assert T_num_M1_N1[2, 0].real == pytest.approx(-_YG0.real, abs=1e-6)

    def test_T21_real(self, T_num_M1_N1):
        # -Yg1(omega[1])*exp(-I*theta) real part ≈ -0.049497
        assert T_num_M1_N1[2, 1].real == pytest.approx(-0.049497, abs=1e-4)

    def test_T21_imag(self, T_num_M1_N1):
        # -Yg1(omega[1])*exp(-I*theta) imag part ≈ +0.021213
        assert T_num_M1_N1[2, 1].imag == pytest.approx(+0.021213, abs=1e-4)

    def test_T22_real(self, T_num_M1_N1):
        # 1 + Yg0*Zs0 + 2*Yg1*Zs1, real part ≈ 1.508
        assert T_num_M1_N1[2, 2].real == pytest.approx(1.508, abs=1e-3)

    def test_T22_imag(self, T_num_M1_N1):
        # 2*Im(Yg1*Zs1) = 2*(0.05*0.05 + 0.02*0.1) = 0.009
        assert T_num_M1_N1[2, 2].imag == pytest.approx(0.009, abs=1e-6)

    def test_T23_real(self, T_num_M1_N1):
        # (Yg0*Zs1 + Yg1*Zs0)*exp(-I*theta) real part ≈ +0.102530
        assert T_num_M1_N1[2, 3].real == pytest.approx(+0.102530, abs=1e-4)

    def test_T23_imag(self, T_num_M1_N1):
        # (Yg0*Zs1 + Yg1*Zs0)*exp(-I*theta) imag part ≈ -0.038891
        assert T_num_M1_N1[2, 3].imag == pytest.approx(-0.038891, abs=1e-4)

    def test_T30_real(self, T_num_M1_N1):
        # -Yg1(omega[0])*exp(I*theta) real part ≈ -0.021213
        assert T_num_M1_N1[3, 0].real == pytest.approx(-0.021213, abs=1e-4)

    def test_T30_imag(self, T_num_M1_N1):
        # -Yg1(omega[0])*exp(I*theta) imag part ≈ -0.049497
        assert T_num_M1_N1[3, 0].imag == pytest.approx(-0.049497, abs=1e-4)

    def test_T31_real(self, T_num_M1_N1):
        assert T_num_M1_N1[3, 1].real == pytest.approx(-_YG0.real, abs=1e-6)

    def test_T32_real(self, T_num_M1_N1):
        # (Yg0*Zs1 + Yg1*Zs0)*exp(+I*theta) real part ≈ +0.038891
        assert T_num_M1_N1[3, 2].real == pytest.approx(+0.038891, abs=1e-4)

    def test_T32_imag(self, T_num_M1_N1):
        # (Yg0*Zs1 + Yg1*Zs0)*exp(+I*theta) imag part ≈ +0.102530
        assert T_num_M1_N1[3, 2].imag == pytest.approx(+0.102530, abs=1e-4)

    def test_T33_real(self, T_num_M1_N1):
        assert T_num_M1_N1[3, 3].real == pytest.approx(1.508, abs=1e-3)

    def test_T33_imag(self, T_num_M1_N1):
        assert T_num_M1_N1[3, 3].imag == pytest.approx(0.009, abs=1e-6)

    def test_M0_determinant_is_one(self, T_num_M0_N0):
        """2x2 matrix with purely real Zs0, Yg0 must have det = 1."""
        det = np.linalg.det(T_num_M0_N0)
        assert det.real == pytest.approx(1.0, abs=1e-10)
        assert det.imag == pytest.approx(0.0, abs=1e-10)

    def test_M0_shape(self, T_num_M0_N0):
        assert T_num_M0_N0.shape == (2, 2)

    def test_M0_T00_is_one(self, T_num_M0_N0):
        assert T_num_M0_N0[0, 0].real == pytest.approx(1.0, abs=1e-10)

    def test_M0_T01_is_minus_Zs0(self, T_num_M0_N0):
        assert T_num_M0_N0[0, 1].real == pytest.approx(-_ZS0.real, abs=1e-10)

    def test_M0_T10_is_minus_Yg0(self, T_num_M0_N0):
        assert T_num_M0_N0[1, 0].real == pytest.approx(-_YG0.real, abs=1e-10)

    def test_M0_T11_is_one_plus_Yg0_Zs0(self, T_num_M0_N0):
        expected = 1.0 + _YG0.real * _ZS0.real
        assert T_num_M0_N0[1, 1].real == pytest.approx(expected, abs=1e-10)


def _make_expected_M2():
    """
    6x6 expected matrix for M=2, ks=[0,1,2] with constant immitance functions.

    Voltage rows pick up Zs harmonics directly (no products).
    Current rows pick up Yg harmonics on V columns, and all convolutions of
    Yg and Zs harmonics on I columns.

    m=1 current-current coupling (alpha1) gains cross terms Yg1*Zs2 + Yg2*Zs1
    relative to M=1 because the m=1 component of Yg*Zs now also gets
    contributions from (m=+2)x(m=-1) and (m=-2)x(m=+1) products.
    """
    e1p = np.exp(+1j * _THETA)
    e1m = np.exp(-1j * _THETA)
    e2p = np.exp(+2j * _THETA)
    e2m = np.exp(-2j * _THETA)
    alpha1 = _YG0 * _ZS1 + _YG1 * _ZS0 + _YG1 * _ZS2 + _YG2 * _ZS1
    alpha2 = _YG0 * _ZS2 + _YG1 * _ZS1 + _YG2 * _ZS0
    dc = 1 + _YG0 * _ZS0 + 2 * _YG1 * _ZS1 + 2 * _YG2 * _ZS2
    return np.array(
        [
            [1, 0, 0, -_ZS0, -_ZS1 * e1m, -_ZS2 * e2m],
            [0, 1, 0, -_ZS1 * e1p, -_ZS0, -_ZS1 * e1m],
            [0, 0, 1, -_ZS2 * e2p, -_ZS1 * e1p, -_ZS0],
            [-_YG0, -_YG1 * e1m, -_YG2 * e2m, dc, alpha1 * e1m, alpha2 * e2m],
            [-_YG1 * e1p, -_YG0, -_YG1 * e1m, alpha1 * e1p, dc, alpha1 * e1m],
            [-_YG2 * e2p, -_YG1 * e1p, -_YG0, alpha2 * e2p, alpha1 * e1p, dc],
        ],
        dtype=complex,
    )


def _make_expected_M1_three_bands():
    """
    6x6 expected matrix for M=1, ks=[0,1,2] with constant immitance functions.

    Direct m=2 terms (Zs2, Yg2) are absent because M=1.
    However the product Yg1·E+1 x Zs1·E+1 inside (1+Yg·Zs)·I still produces a
    second-harmonic current coupling (beta2 = Yg1·Zs1), linking non-adjacent
    sidebands k=0 and k=2.  The m=2 I-I coefficient at M=2 (alpha2) gains the
    additional Yg0·Zs2 and Yg2·Zs0 terms absent here.
    """
    e1p = np.exp(+1j * _THETA)
    e1m = np.exp(-1j * _THETA)
    e2p = np.exp(+2j * _THETA)
    e2m = np.exp(-2j * _THETA)
    beta1 = _YG0 * _ZS1 + _YG1 * _ZS0  # m=1 I-I: no Yg1*Zs2+Yg2*Zs1 cross terms
    beta2 = _YG1 * _ZS1  # m=2 I-I: product term only (no Yg0*Zs2+Yg2*Zs0)
    dc = 1 + _YG0 * _ZS0 + 2 * _YG1 * _ZS1  # no Yg2*Zs2 term
    return np.array(
        [
            [1, 0, 0, -_ZS0, -_ZS1 * e1m, 0],
            [0, 1, 0, -_ZS1 * e1p, -_ZS0, -_ZS1 * e1m],
            [0, 0, 1, 0, -_ZS1 * e1p, -_ZS0],
            [-_YG0, -_YG1 * e1m, 0, dc, beta1 * e1m, beta2 * e2m],
            [-_YG1 * e1p, -_YG0, -_YG1 * e1m, beta1 * e1p, dc, beta1 * e1m],
            [0, -_YG1 * e1p, -_YG0, beta2 * e2p, beta1 * e1p, dc],
        ],
        dtype=complex,
    )


@pytest.fixture(scope="module")
def T_num_M2_N2():
    cell_obj = CellSingleMode()
    M, ks_state = 2, [0, 1, 2]
    T_sym, state_syms, Zs_m, Yg_m = cell_obj.build_symbolic_transfer_matrix(M, ks_state)
    dim = len(state_syms)
    return cell_obj.build_numeric_matrix(
        T_sym,
        dim,
        M,
        ks_state,
        Zs_m,
        Yg_m,
        _make_cell(),
        _omega_val,
        _OMEGA_P,
        k_val=0,
    )


class TestNumericalMatrixM2:
    def test_shape(self, T_num_M2_N2):
        assert T_num_M2_N2.shape == (6, 6)

    def test_full_matrix_matches_formula(self, T_num_M2_N2):
        """All 36 entries verified against hand-derived analytical formula."""
        np.testing.assert_allclose(T_num_M2_N2, _make_expected_M2(), atol=1e-10)

    # --- entries that are NEW in M=2 vs M=1 ---

    def test_T05_m2_zs_voltage_coupling(self, T_num_M2_N2):
        """T[0,5] = -Zs2*exp(-2I*theta): absent in M=1."""
        assert T_num_M2_N2[0, 5] == pytest.approx(
            -_ZS2 * np.exp(-2j * _THETA), abs=1e-8
        )

    def test_T23_m2_zs_voltage_coupling(self, T_num_M2_N2):
        """T[2,3] = -Zs2*exp(+2I*theta): conjugate of T[0,5]."""
        assert T_num_M2_N2[2, 3] == pytest.approx(
            -_ZS2 * np.exp(+2j * _THETA), abs=1e-8
        )

    def test_T32_m2_yg_current_coupling(self, T_num_M2_N2):
        """T[3,2] = -Yg2*exp(-2I*theta): absent in M=1."""
        assert T_num_M2_N2[3, 2] == pytest.approx(
            -_YG2 * np.exp(-2j * _THETA), abs=1e-8
        )

    def test_T50_m2_yg_current_coupling(self, T_num_M2_N2):
        """T[5,0] = -Yg2*exp(+2I*theta): conjugate of T[3,2]."""
        assert T_num_M2_N2[5, 0] == pytest.approx(
            -_YG2 * np.exp(+2j * _THETA), abs=1e-8
        )

    def test_T35_m2_ii_coupling(self, T_num_M2_N2):
        """T[3,5] = alpha2*exp(-2I*theta) where alpha2 = Yg0*Zs2 + Yg1*Zs1 + Yg2*Zs0."""
        alpha2 = _YG0 * _ZS2 + _YG1 * _ZS1 + _YG2 * _ZS0
        assert T_num_M2_N2[3, 5] == pytest.approx(
            alpha2 * np.exp(-2j * _THETA), abs=1e-8
        )

    def test_T53_m2_ii_coupling(self, T_num_M2_N2):
        """T[5,3] = alpha2*exp(+2I*theta)."""
        alpha2 = _YG0 * _ZS2 + _YG1 * _ZS1 + _YG2 * _ZS0
        assert T_num_M2_N2[5, 3] == pytest.approx(
            alpha2 * np.exp(+2j * _THETA), abs=1e-8
        )

    def test_T34_m1_ii_coupling_gains_cross_terms(self, T_num_M2_N2):
        """T[3,4] = alpha1*exp(-I*theta): alpha1 gains Yg1*Zs2+Yg2*Zs1 vs M=1."""
        alpha1 = _YG0 * _ZS1 + _YG1 * _ZS0 + _YG1 * _ZS2 + _YG2 * _ZS1
        assert T_num_M2_N2[3, 4] == pytest.approx(
            alpha1 * np.exp(-1j * _THETA), abs=1e-8
        )

    def test_current_diagonal_gains_Yg2_Zs2(self, T_num_M2_N2):
        """Diagonal = 1 + Yg0*Zs0 + 2*Yg1*Zs1 + 2*Yg2*Zs2 (new Yg2*Zs2 term vs M=1)."""
        dc = 1 + _YG0 * _ZS0 + 2 * _YG1 * _ZS1 + 2 * _YG2 * _ZS2
        for i in [3, 4, 5]:
            assert T_num_M2_N2[i, i] == pytest.approx(dc, abs=1e-8)


@pytest.fixture(scope="module")
def T_num_M1_three_bands():
    """M=1, ks=[0,1,2]: 6x6 matrix; m=2 coupling positions must be zero."""
    cell_obj = CellSingleMode()
    M, ks_state = 1, [0, 1, 2]
    T_sym, state_syms, Zs_m, Yg_m = cell_obj.build_symbolic_transfer_matrix(M, ks_state)
    dim = len(state_syms)
    return cell_obj.build_numeric_matrix(
        T_sym,
        dim,
        M,
        ks_state,
        Zs_m,
        Yg_m,
        _make_cell(),
        _omega_val,
        _OMEGA_P,
        k_val=0,
    )


class TestNumericalMatrixM1ThreeBands:
    def test_shape(self, T_num_M1_three_bands):
        assert T_num_M1_three_bands.shape == (6, 6)

    def test_full_matrix_matches_formula(self, T_num_M1_three_bands):
        """All 36 entries verified against hand-derived formula."""
        np.testing.assert_allclose(
            T_num_M1_three_bands, _make_expected_M1_three_bands(), atol=1e-10
        )

    def test_m2_voltage_coupling_absent(self, T_num_M1_three_bands):
        """T[0,5] = T[2,3] = 0: M=1 truncation forbids m=2 Zs coupling."""
        assert abs(T_num_M1_three_bands[0, 5]) == pytest.approx(0, abs=1e-12)
        assert abs(T_num_M1_three_bands[2, 3]) == pytest.approx(0, abs=1e-12)

    def test_m2_current_voltage_coupling_absent(self, T_num_M1_three_bands):
        """T[3,2] = T[5,0] = 0: M=1 truncation forbids m=2 Yg coupling."""
        assert abs(T_num_M1_three_bands[3, 2]) == pytest.approx(0, abs=1e-12)
        assert abs(T_num_M1_three_bands[5, 0]) == pytest.approx(0, abs=1e-12)

    def test_m2_ii_cascade_coupling_is_yg1_zs1_product(self, T_num_M1_three_bands):
        """
        T[3,5] = Yg1*Zs1*exp(-2I*theta): the Yg1·E+1 x Zs1·E+1 product inside
        (1+Yg·Zs)·I generates a second-harmonic I-I coupling linking non-adjacent
        sidebands k=0 and k=2 even at M=1.  At M=2 this becomes alpha2*exp(-2I*theta)
        where alpha2 = Yg0*Zs2 + Yg1*Zs1 + Yg2*Zs0 gains the direct Zs2/Yg2 terms.
        """
        beta2 = _YG1 * _ZS1
        assert T_num_M1_three_bands[3, 5] == pytest.approx(
            beta2 * np.exp(-2j * _THETA), abs=1e-10
        )
        assert T_num_M1_three_bands[5, 3] == pytest.approx(
            beta2 * np.exp(+2j * _THETA), abs=1e-10
        )

    def test_voltage_block_is_identity(self, T_num_M1_three_bands):
        for i in range(3):
            assert T_num_M1_three_bands[i, i].real == pytest.approx(1, abs=1e-10)
            for j in range(3):
                if i != j:
                    assert abs(T_num_M1_three_bands[i, j]) == pytest.approx(
                        0, abs=1e-10
                    )

    def test_direct_zs0_couplings(self, T_num_M1_three_bands):
        """T[k, k+3] = -Zs0 for each sideband k."""
        for k in range(3):
            assert T_num_M1_three_bands[k, k + 3].real == pytest.approx(
                -_ZS0.real, abs=1e-8
            )

    def test_m1_voltage_coupling_non_adjacent_bands(self, T_num_M1_three_bands):
        """T[0,4] and T[2,4] equal -Zs1*exp(-I*theta); T[1,3] and T[1,5] equal -Zs1*exp(+I*theta)."""
        zm = -_ZS1 * np.exp(-1j * _THETA)
        zp = -_ZS1 * np.exp(+1j * _THETA)
        assert T_num_M1_three_bands[0, 4] == pytest.approx(zm, abs=1e-8)
        assert T_num_M1_three_bands[2, 4] == pytest.approx(zp, abs=1e-8)
        assert T_num_M1_three_bands[1, 3] == pytest.approx(zp, abs=1e-8)
        assert T_num_M1_three_bands[1, 5] == pytest.approx(zm, abs=1e-8)

    def test_m1_current_cross_coupling_no_extra_terms(self, T_num_M1_three_bands):
        """alpha1 = Yg0*Zs1+Yg1*Zs0 only (no Yg1*Zs2+Yg2*Zs1 cross terms at M=1)."""
        beta1 = _YG0 * _ZS1 + _YG1 * _ZS0
        assert T_num_M1_three_bands[3, 4] == pytest.approx(
            beta1 * np.exp(-1j * _THETA), abs=1e-8
        )
        assert T_num_M1_three_bands[4, 3] == pytest.approx(
            beta1 * np.exp(+1j * _THETA), abs=1e-8
        )


_OMEGA_S_SWEEP = np.array([0.8, 1.0, 1.2, 1.4])

# Cells with frequency-dependent Zs0/Yg0 so T_grid varies across freq axis.
_CELL_FREQ_PARAMS = [
    CellImmitance(
        theta=np.pi / 4,
        Zs0_fn=lambda omega: 1.0 + 0.1 * omega,
        Yg0_fn=lambda omega: 0.5 + 0.05 * omega,
        Zs_harm_fn=lambda m, omega: {1: 0.10 + 0.05j}.get(m, 0j),
        Yg_harm_fn=lambda m, omega: {1: 0.05 + 0.02j}.get(m, 0j),
    ),
    CellImmitance(
        theta=np.pi / 4,
        Zs0_fn=lambda omega: 1.5 + 0.1 * omega,
        Yg0_fn=lambda omega: 0.3 + 0.03 * omega,
        Zs_harm_fn=lambda m, omega: {1: 0.20 + 0.03j}.get(m, 0j),
        Yg_harm_fn=lambda m, omega: {1: 0.08 + 0.01j}.get(m, 0j),
    ),
]


@pytest.fixture(scope="module")
def cell_freq_fixture():
    """Return (cell_obj, T_sym, dim, M, ks_state, Zs_m, Yg_m, T_grid)."""
    cell_obj = CellSingleMode()
    M, ks_state = 1, [0, 1]
    T_sym, state_syms, Zs_m, Yg_m = cell_obj.build_symbolic_transfer_matrix(M, ks_state)
    dim = len(state_syms)
    T_grid = cell_obj.build_cell_freq_matrices(
        T_sym,
        dim,
        M,
        ks_state,
        Zs_m,
        Yg_m,
        _OMEGA_S_SWEEP,
        _OMEGA_P,
        _CELL_FREQ_PARAMS,
    )
    return cell_obj, T_sym, dim, M, ks_state, Zs_m, Yg_m, T_grid


class TestCellFreqMatrices:
    def test_shape(self, cell_freq_fixture):
        *_, T_grid = cell_freq_fixture
        assert T_grid.shape == (len(_OMEGA_S_SWEEP), len(_CELL_FREQ_PARAMS), 4, 4)

    def test_dtype_is_complex(self, cell_freq_fixture):
        *_, T_grid = cell_freq_fixture
        assert np.issubdtype(T_grid.dtype, np.complexfloating)

    def test_each_entry_matches_build_numeric_matrix(self, cell_freq_fixture):
        """T_grid[f, c] must equal the individually computed matrix."""
        cell_obj, T_sym, dim, M, ks_state, Zs_m, Yg_m, T_grid = cell_freq_fixture
        for f, os in enumerate(_OMEGA_S_SWEEP):
            omega_val_fn = lambda k, _os=os: _os + k * _OMEGA_P
            for c, cell_params in enumerate(_CELL_FREQ_PARAMS):
                T_ref = cell_obj.build_numeric_matrix(
                    T_sym,
                    dim,
                    M,
                    ks_state,
                    Zs_m,
                    Yg_m,
                    cell_params,
                    omega_val_fn,
                    _OMEGA_P,
                    k_val=0,
                )
                np.testing.assert_allclose(T_grid[f, c], T_ref, atol=1e-12)

    def test_different_cells_give_different_matrices(self, cell_freq_fixture):
        """Cells with different Zs/Yg must produce different matrices."""
        *_, T_grid = cell_freq_fixture
        assert not np.allclose(T_grid[0, 0], T_grid[0, 1])

    def test_different_freqs_give_different_matrices(self, cell_freq_fixture):
        """Different signal frequencies must produce different matrices."""
        *_, T_grid = cell_freq_fixture
        assert not np.allclose(T_grid[0, 0], T_grid[1, 0])

    def test_freq_axis_independent_of_cell_axis(self, cell_freq_fixture):
        """Cell ordering is consistent across all frequency slices."""
        *_, T_grid = cell_freq_fixture
        assert not np.allclose(T_grid[1, 0], T_grid[1, 1])
