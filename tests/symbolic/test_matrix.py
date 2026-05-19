"""
Unit tests for src/symbolic/matrix.py.

Test groups
-----------
1. fourier_basis          — algebraic identities on E(j)
2. series structure        — Zs/Yg series at M=0 and M=1
3. harmonic extraction     — DC and first-order coefficients at M=1
4. symbolic transfer matrix — structure and determinant at M=0,N=0
5. numerical evaluation    — values match the notebook reference output
"""
import numpy as np
import pytest
from sympy import (
    exp, I, simplify, expand, symbols, Integer, Function,
    Eq, zeros, Symbol
)

from src.symbolic.matrix import (
    fourier_basis,
    make_harmonic_functions,
    build_Zs_series,
    build_Yg_shunt,
    extract_harmonic_results,
    build_transfer_matrix,
    build_symbolic_transfer_matrix,
    build_numeric_matrix,
    # module-level symbols
    t, theta, omega_p, k, n, omega, Zs0, Yg0, V, Ic, xi,
)


class TestFourierBasis:
    def test_zero_harmonic_is_unity(self):
        assert fourier_basis(0) == 1

    def test_positive_harmonic(self):
        assert fourier_basis(1) == exp(I * omega_p * t)

    def test_conjugate_pair_product_is_unity(self):
        product = expand(fourier_basis(2) * fourier_basis(-2))
        assert simplify(product) == 1

    def test_additivity_of_exponents(self):
        # E(j1) * E(j2) == E(j1+j2)
        assert simplify(fourier_basis(1) * fourier_basis(2) - fourier_basis(3)) == 0

    def test_negative_is_conjugate_of_positive(self):
        # E(-j) = conj(E(j)) when omega_p and t are real
        diff = simplify(fourier_basis(-1) - exp(-I * omega_p * t))
        assert diff == 0


class TestSeriesStructure:
    def test_Zs_series_M0_is_dc_only(self):
        """With no harmonic functions (M=0), Zs(t) reduces to the DC term Zs0."""
        Zs_t = build_Zs_series([])
        assert Zs_t == Zs0

    def test_Yg_series_M0_is_dc_only(self):
        Yg_t = build_Yg_shunt([])
        assert Yg_t == Yg0

    def test_Zs_series_M1_contains_Zs1(self):
        Zs_m, _ = make_harmonic_functions(1)
        Zs_t = build_Zs_series(Zs_m)
        Zs1 = Function('Zs1')
        # The xi-valued Zs1 must appear in the series
        assert Zs1(xi) in Zs_t.free_symbols or Zs_t.has(Zs1(xi))

    def test_Yg_series_M1_contains_Yg1(self):
        _, Yg_m = make_harmonic_functions(1)
        Yg_t = build_Yg_shunt(Yg_m)
        Yg1 = Function('Yg1')
        assert Yg_t.has(Yg1(xi))

    def test_Zs_series_M1_has_three_terms(self):
        """DC + E(+1) term + E(-1) term -> 3 additive pieces after expansion."""
        Zs_m, _ = make_harmonic_functions(1)
        Zs_t = expand(build_Zs_series(Zs_m))
        assert len(Zs_t.as_ordered_terms()) == 3

    def test_Zs_series_M1_DC_coefficient_is_Zs0(self):
        """The coefficient of E(0)=1 inside Zs(t) is just Zs0."""
        Zs_m, _ = make_harmonic_functions(1)
        Zs_t = expand(build_Zs_series(Zs_m))
        dc = Zs_t.coeff(fourier_basis(1))   # coeff of E(1) — should not be Zs0
        # The constant (non-exponential) part is Zs0
        dc_part = Zs_t - Zs_t.coeff(fourier_basis(1)) * fourier_basis(1) \
                        - Zs_t.coeff(fourier_basis(-1)) * fourier_basis(-1)
        assert simplify(dc_part - Zs0) == 0


class TestHarmonicExtraction:
    @pytest.fixture(scope='class')
    def results_M0(self):
        return extract_harmonic_results(0, build_Zs_series([]), build_Yg_shunt([]))

    @pytest.fixture(scope='class')
    def results_M1(self):
        Zs_m, Yg_m = make_harmonic_functions(1)
        return extract_harmonic_results(1, build_Zs_series(Zs_m), build_Yg_shunt(Yg_m))

    def test_M0_only_dc_key(self, results_M0):
        """At M=0 (no pump harmonics) only the DC key j=0 should be present."""
        assert set(results_M0.keys()) == {0}

    def test_M1_has_keys_minus2_to_plus2(self, results_M1):
        assert set(results_M1.keys()) == {-2, -1, 0, 1, 2}

    def test_M0_dc_voltage(self, results_M0):
        """V[k,n+1] DC = V[k,n] - Zs0*I[k,n]."""
        cv0, _ = results_M0[0]
        expected = V[k, n] - Zs0 * Ic[k, n]
        assert simplify(expand(cv0 - expected)) == 0

    def test_M0_dc_current(self, results_M0):
        """I[k,n+1] DC = -Yg0*V[k,n] + (1 + Yg0*Zs0)*I[k,n]."""
        _, ci0 = results_M0[0]
        expected = -Yg0 * V[k, n] + (1 + Yg0 * Zs0) * Ic[k, n]
        assert simplify(expand(ci0 - expected)) == 0

    def test_M1_j1_voltage_coeff_involves_Zs1(self, results_M1):
        """The j=+1 voltage harmonic must contain a Zs1 function."""
        cv1, _ = results_M1[1]
        Zs1 = Function('Zs1')
        assert cv1.has(Zs1)

    def test_M1_dc_has_no_exponentials(self, results_M1):
        """The DC result must not contain any Floquet exponentials."""
        cv0, ci0 = results_M1[0]
        for j in [-2, -1, 1, 2]:
            assert cv0.coeff(fourier_basis(j)) == 0
            assert ci0.coeff(fourier_basis(j)) == 0


class TestSymbolicTransferMatrix:
    @pytest.fixture(scope='class')
    def T_M0_N0(self):
        """2×2 transfer matrix for M=0 (no pump), single mode k=0."""
        T_sym, state_syms, Zs_m, Yg_m = build_symbolic_transfer_matrix(
            M=0, ks_state=[0]
        )
        return T_sym, state_syms

    @pytest.fixture(scope='class')
    def T_M1_N1(self):
        """4×4 transfer matrix for M=1, sidebands [0,1]."""
        T_sym, state_syms, Zs_m, Yg_m = build_symbolic_transfer_matrix(
            M=1, ks_state=[0, 1]
        )
        return T_sym, state_syms

    # ── shape ────────────────────────────────────────────────────────────────

    def test_shape_M0_N0(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert T_sym.shape == (2, 2)

    def test_shape_M1_N1(self, T_M1_N1):
        T_sym, _ = T_M1_N1
        assert T_sym.shape == (4, 4)

    # ── M=0 entries ──────────────────────────────────────────────────────────

    def test_M0_T00_is_one(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert T_sym[0, 0] == 1

    def test_M0_T01_is_minus_Zs0(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert simplify(T_sym[0, 1] + Zs0) == 0

    def test_M0_T10_is_minus_Yg0(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert simplify(T_sym[1, 0] + Yg0) == 0

    def test_M0_T11_is_one_plus_Yg0_Zs0(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert simplify(T_sym[1, 1] - (1 + Yg0 * Zs0)) == 0

    def test_M0_determinant_is_one(self, T_M0_N0):
        """Reciprocal network: det(T) = AD - BC = 1."""
        T_sym, _ = T_M0_N0
        det = T_sym[0, 0] * T_sym[1, 1] - T_sym[0, 1] * T_sym[1, 0]
        assert simplify(expand(det) - 1) == 0

    # ── M=1 structure ─────────────────────────────────────────────────────────

    def test_M1_diagonal_voltage_rows_are_one(self, T_M1_N1):
        """T[0,0] and T[2,2] are both 1 (V[j,n] -> V[j,n+1] direct coupling)."""
        T_sym, _ = T_M1_N1
        assert T_sym[0, 0] == 1
        assert T_sym[2, 2] == 1

    def test_M1_voltage_rows_have_no_V_cross_coupling(self, T_M1_N1):
        """V[0,n+1] does not depend on V[1,n] (T[0,2] = 0)."""
        T_sym, _ = T_M1_N1
        assert T_sym[0, 2] == 0

    def test_M1_T01_is_minus_Zs0(self, T_M1_N1):
        T_sym, _ = T_M1_N1
        assert simplify(T_sym[0, 1] + Zs0) == 0

    def test_M1_T23_is_minus_Zs0(self, T_M1_N1):
        """Same -Zs0 appears in the second voltage row."""
        T_sym, _ = T_M1_N1
        assert simplify(T_sym[2, 3] + Zs0) == 0

    def test_M1_current_row0_no_Zs2_terms(self, T_M1_N1):
        """With M=1, Zs2 should not appear anywhere in the matrix."""
        T_sym, _ = T_M1_N1
        Zs2 = Function('Zs2')
        assert not T_sym.has(Zs2)

    def test_state_syms_ordering(self, T_M1_N1):
        """State vector is interleaved: [V[0,n], I[0,n], V[1,n], I[1,n]]."""
        _, state_syms = T_M1_N1
        assert state_syms[0] == V[0, n]
        assert state_syms[1] == Ic[0, n]
        assert state_syms[2] == V[1, n]
        assert state_syms[3] == Ic[1, n]


_OMEGA_S  = 1.0
_OMEGA_P  = 0.1
_THETA    = np.pi / 4
_ZS0      = 1.0 + 0.0j
_YG0      = 0.5 + 0.0j


def _omega_val(q):
    return _OMEGA_S + q * _OMEGA_P


def _Zs_num(m, freq):
    return {1: 0.1 + 0.05j, 2: 0.02 + 0.01j}.get(m, 0 + 0j)


def _Yg_num(m, freq):
    return {1: 0.05 + 0.02j, 2: 0.01 + 0.005j}.get(m, 0 + 0j)


@pytest.fixture(scope='module')
def T_num_M1_N1():
    M, js_state = 1, [0, 1]
    T_sym, state_syms, Zs_m, Yg_m = build_symbolic_transfer_matrix(M, js_state)
    dim = len(state_syms)
    return build_numeric_matrix(
        T_sym, dim, M, js_state, Zs_m, Yg_m,
        _ZS0, _YG0, _THETA, _OMEGA_P,
        _omega_val, _Zs_num, _Yg_num,
        k_val=0,
    )


@pytest.fixture(scope='module')
def T_num_M0_N0():
    M, js_state = 0, [0]
    T_sym, state_syms, Zs_m, Yg_m = build_symbolic_transfer_matrix(M, js_state)
    dim = len(state_syms)
    return build_numeric_matrix(
        T_sym, dim, M, js_state, Zs_m, Yg_m,
        _ZS0, _YG0, _THETA, _OMEGA_P,
        _omega_val, _Zs_num, _Yg_num,
        k_val=0,
    )


class TestNumericalMatrix:
    def test_shape(self, T_num_M1_N1):
        assert T_num_M1_N1.shape == (4, 4)

    def test_dtype_is_complex(self, T_num_M1_N1):
        assert np.issubdtype(T_num_M1_N1.dtype, np.complexfloating)

    # ── Specific real-part values from the notebook output ───────────────────

    def test_T00_real_is_one(self, T_num_M1_N1):
        assert T_num_M1_N1[0, 0].real == pytest.approx(1.0, abs=1e-10)

    def test_T01_real_is_minus_Zs0(self, T_num_M1_N1):
        assert T_num_M1_N1[0, 1].real == pytest.approx(-_ZS0.real, abs=1e-6)

    def test_T02_is_zero(self, T_num_M1_N1):
        assert abs(T_num_M1_N1[0, 2]) == pytest.approx(0.0, abs=1e-10)

    def test_T03_real(self, T_num_M1_N1):
        # -Zs1(omega[1])*exp(-I*theta) real part ≈ -0.106066
        assert T_num_M1_N1[0, 3].real == pytest.approx(-0.106066, abs=1e-4)

    def test_T10_real(self, T_num_M1_N1):
        assert T_num_M1_N1[1, 0].real == pytest.approx(-_YG0.real, abs=1e-6)

    def test_T11_real(self, T_num_M1_N1):
        # Yg0*Zs0 + 2*Yg1(omega[0])*Zs1(omega[0]) + 1, real part ≈ 1.508
        assert T_num_M1_N1[1, 1].real == pytest.approx(1.508, abs=1e-3)

    def test_T21_real(self, T_num_M1_N1):
        # -Zs1(omega[0])*exp(I*theta) real part ≈ -0.035355
        assert T_num_M1_N1[2, 1].real == pytest.approx(-0.035355, abs=1e-4)

    def test_T22_real_is_one(self, T_num_M1_N1):
        assert T_num_M1_N1[2, 2].real == pytest.approx(1.0, abs=1e-10)

    def test_T23_real_is_minus_Zs0(self, T_num_M1_N1):
        assert T_num_M1_N1[2, 3].real == pytest.approx(-_ZS0.real, abs=1e-6)

    def test_T30_real(self, T_num_M1_N1):
        # -Yg1(omega[0])*exp(I*theta) real part ≈ -0.021213
        assert T_num_M1_N1[3, 0].real == pytest.approx(-0.021213, abs=1e-4)

    def test_T32_real(self, T_num_M1_N1):
        assert T_num_M1_N1[3, 2].real == pytest.approx(-_YG0.real, abs=1e-6)

    def test_T33_real(self, T_num_M1_N1):
        assert T_num_M1_N1[3, 3].real == pytest.approx(1.508, abs=1e-3)

    # ── Reciprocity: det(T) = 1 for lossless M=0 case ───────────────────────

    def test_M0_determinant_is_one(self, T_num_M0_N0):
        """2×2 matrix with purely real Zs0, Yg0 must have det = 1."""
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
