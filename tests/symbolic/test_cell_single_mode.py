import numpy as np
import pytest
from sympy import exp, I, simplify, expand, Function

from src.symbolic.cell_single_mode import CellSingleMode


def _setup_cell(cls):
    """Attach a fresh CellSingleMode instance's methods and symbols to a test class."""
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
        # E(j1) * E(j2) == E(j1+j2)
        assert (
            simplify(
                self.fourier_basis(1) * self.fourier_basis(2) - self.fourier_basis(3)
            )
            == 0
        )

    def test_negative_is_conjugate_of_positive(self):
        # E(-j) = conj(E(j)) when omega_p and t are real
        diff = simplify(self.fourier_basis(-1) - exp(-I * self.omega_p * self.t))
        assert diff == 0


class TestSeriesStructure:
    @classmethod
    def setup_class(cls):
        _setup_cell(cls)

    def test_Zs_series_M0_is_dc_only(self):
        """With no harmonic functions (M=0), Zs(t) reduces to the DC term Zs0."""
        assert self.build_Zs_series([]) == self.Zs0

    def test_Yg_series_M0_is_dc_only(self):
        assert self.build_Yg_shunt([]) == self.Yg0

    def test_Zs_series_M1_contains_Zs1(self):
        Zs_m, _ = self.make_harmonic_functions(1)
        Zs_t = self.build_Zs_series(Zs_m)
        Zs1 = Function("Zs1")
        assert Zs1(self.xi) in Zs_t.free_symbols or Zs_t.has(Zs1(self.xi))

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
        """The coefficient of E(0)=1 inside Zs(t) is just Zs0."""
        Zs_m, _ = self.make_harmonic_functions(1)
        Zs_t = expand(self.build_Zs_series(Zs_m))
        dc_part = (
            Zs_t
            - Zs_t.coeff(self.fourier_basis(1)) * self.fourier_basis(1)
            - Zs_t.coeff(self.fourier_basis(-1)) * self.fourier_basis(-1)
        )
        assert simplify(dc_part - self.Zs0) == 0


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
        expected = self.V[self.k, self.n] - self.Zs0 * self.Ic[self.k, self.n]
        assert simplify(expand(cv0 - expected)) == 0

    def test_M0_dc_current(self, results_M0):
        """I[k,n+1] DC = -Yg0*V[k,n] + (1 + Yg0*Zs0)*I[k,n]."""
        _, ci0 = results_M0[0]
        expected = (
            -self.Yg0 * self.V[self.k, self.n]
            + (1 + self.Yg0 * self.Zs0) * self.Ic[self.k, self.n]
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
        assert simplify(T_sym[0, 1] + self.Zs0) == 0

    def test_M0_T10_is_minus_Yg0(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert simplify(T_sym[1, 0] + self.Yg0) == 0

    def test_M0_T11_is_one_plus_Yg0_Zs0(self, T_M0_N0):
        T_sym, _ = T_M0_N0
        assert simplify(T_sym[1, 1] - (1 + self.Yg0 * self.Zs0)) == 0

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
        """V[0,n+1] coefficient of I[0,n] is -Zs0 (T[0,2])."""
        T_sym, _ = T_M1_N1
        assert simplify(T_sym[0, 2] + self.Zs0) == 0

    def test_M1_T13_is_minus_Zs0(self, T_M1_N1):
        """Same -Zs0 appears in the second voltage row (T[1,3])."""
        T_sym, _ = T_M1_N1
        assert simplify(T_sym[1, 3] + self.Zs0) == 0

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


_OMEGA_S = 1.0
_OMEGA_P = 0.1
_THETA = np.pi / 4
_ZS0 = 1.0 + 0.0j
_YG0 = 0.5 + 0.0j


def _omega_val(q):
    return _OMEGA_S + q * _OMEGA_P


def _Zs_num(m, omega):
    return {1: 0.1 + 0.05j, 2: 0.02 + 0.01j}.get(m, 0 + 0j)


def _Yg_num(m, omega):
    return {1: 0.05 + 0.02j, 2: 0.01 + 0.005j}.get(m, 0 + 0j)


@pytest.fixture(scope="module")
def T_num_M1_N1():
    cell = CellSingleMode()
    M, ks_state = 1, [0, 1]
    T_sym, state_syms, Zs_m, Yg_m = cell.build_symbolic_transfer_matrix(M, ks_state)
    dim = len(state_syms)
    return cell.build_numeric_matrix(
        T_sym,
        dim,
        M,
        ks_state,
        Zs_m,
        Yg_m,
        lambda omega: _ZS0,
        lambda omega: _YG0,
        _THETA,
        _OMEGA_P,
        _omega_val,
        _Zs_num,
        _Yg_num,
        k_val=0,
    )


@pytest.fixture(scope="module")
def T_num_M0_N0():
    cell = CellSingleMode()
    M, ks_state = 0, [0]
    T_sym, state_syms, Zs_m, Yg_m = cell.build_symbolic_transfer_matrix(M, ks_state)
    dim = len(state_syms)
    return cell.build_numeric_matrix(
        T_sym,
        dim,
        M,
        ks_state,
        Zs_m,
        Yg_m,
        lambda omega: _ZS0,
        lambda omega: _YG0,
        _THETA,
        _OMEGA_P,
        _omega_val,
        _Zs_num,
        _Yg_num,
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
        # V[0,n+1] has no direct coupling to V[1,n]
        assert abs(T_num_M1_N1[0, 1]) == pytest.approx(0.0, abs=1e-10)

    def test_T02_real_is_minus_Zs0(self, T_num_M1_N1):
        # V[0,n+1] = ... - Zs0*I[0,n] - ...
        assert T_num_M1_N1[0, 2].real == pytest.approx(-_ZS0.real, abs=1e-6)

    def test_T03_real(self, T_num_M1_N1):
        # -Zs1(omega[1])*exp(-I*theta) real part ≈ -0.106066
        assert T_num_M1_N1[0, 3].real == pytest.approx(-0.106066, abs=1e-4)

    def test_T11_real_is_one(self, T_num_M1_N1):
        assert T_num_M1_N1[1, 1].real == pytest.approx(1.0, abs=1e-10)

    def test_T12_real(self, T_num_M1_N1):
        # -Zs1(omega[0])*exp(I*theta) real part ≈ -0.035355
        assert T_num_M1_N1[1, 2].real == pytest.approx(-0.035355, abs=1e-4)

    def test_T13_real_is_minus_Zs0(self, T_num_M1_N1):
        assert T_num_M1_N1[1, 3].real == pytest.approx(-_ZS0.real, abs=1e-6)

    def test_T20_real(self, T_num_M1_N1):
        # I[0,n+1] = -Yg0*V[0,n+1] + ... → -Yg0 coefficient on V[0,n]
        assert T_num_M1_N1[2, 0].real == pytest.approx(-_YG0.real, abs=1e-6)

    def test_T22_real(self, T_num_M1_N1):
        # Yg0*Zs0 + 2*Yg1(omega[0])*Zs1(omega[0]) + 1, real part ≈ 1.508
        assert T_num_M1_N1[2, 2].real == pytest.approx(1.508, abs=1e-3)

    def test_T30_real(self, T_num_M1_N1):
        # -Yg1(omega[0])*exp(I*theta) real part ≈ -0.021213
        assert T_num_M1_N1[3, 0].real == pytest.approx(-0.021213, abs=1e-4)

    def test_T31_real(self, T_num_M1_N1):
        assert T_num_M1_N1[3, 1].real == pytest.approx(-_YG0.real, abs=1e-6)

    def test_T33_real(self, T_num_M1_N1):
        assert T_num_M1_N1[3, 3].real == pytest.approx(1.508, abs=1e-3)

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


_OMEGA_S_SWEEP = [0.8, 1.0, 1.2, 1.4]  # Nf = 4 signal frequencies
_THETA_FOR_FREQ = [0.0, np.pi / 6, np.pi / 3, np.pi / 2]  # one theta per freq point

# Two cells with distinct impedance/admittance profiles
_CELL_PARAMS = [
    dict(
        Zs0_fn=lambda omega: 1.0 + 0.0j,
        Yg0_fn=lambda omega: 0.5 + 0.0j,
        Zs_num_fn=lambda m, _: {1: 0.10 + 0.05j}.get(m, 0j),
        Yg_num_fn=lambda m, _: {1: 0.05 + 0.02j}.get(m, 0j),
    ),
    dict(
        Zs0_fn=lambda omega: 1.5 + 0.0j,
        Yg0_fn=lambda omega: 0.3 + 0.0j,
        Zs_num_fn=lambda m, _: {1: 0.20 + 0.03j}.get(m, 0j),
        Yg_num_fn=lambda m, _: {1: 0.08 + 0.01j}.get(m, 0j),
    ),
]


def _make_freq_params(omega_s, theta=_THETA):
    return dict(
        omega_val_fn=lambda k, os=omega_s: os + k * _OMEGA_P,
        omega_p_val=_OMEGA_P,
        theta_val=theta,
        k_val=0,
    )


@pytest.fixture(scope="module")
def cell_freq_fixture():
    """Return (cell, T_sym, dim, Zs_m, Yg_m, T_grid)."""
    cell = CellSingleMode()
    M, ks_state = 1, [0, 1]
    T_sym, state_syms, Zs_m, Yg_m = cell.build_symbolic_transfer_matrix(M, ks_state)
    dim = len(state_syms)
    freq_params_list = [
        _make_freq_params(os, th) for os, th in zip(_OMEGA_S_SWEEP, _THETA_FOR_FREQ)
    ]
    T_grid = cell.build_cell_freq_matrices(
        T_sym, dim, M, ks_state, Zs_m, Yg_m, _CELL_PARAMS, freq_params_list
    )
    return cell, T_sym, dim, M, ks_state, Zs_m, Yg_m, T_grid


class TestCellFreqMatrices:
    def test_shape(self, cell_freq_fixture):
        *_, T_grid = cell_freq_fixture
        assert T_grid.shape == (len(_OMEGA_S_SWEEP), len(_CELL_PARAMS), 4, 4)

    def test_dtype_is_complex(self, cell_freq_fixture):
        *_, T_grid = cell_freq_fixture
        assert np.issubdtype(T_grid.dtype, np.complexfloating)

    def test_each_entry_matches_build_numeric_matrix(self, cell_freq_fixture):
        """T_grid[f, c] must equal the individually computed matrix."""
        cell, T_sym, dim, M, ks_state, Zs_m, Yg_m, T_grid = cell_freq_fixture
        for f, (os, th) in enumerate(zip(_OMEGA_S_SWEEP, _THETA_FOR_FREQ)):
            freq_p = _make_freq_params(os, th)
            for c, cell_p in enumerate(_CELL_PARAMS):
                T_ref = cell.build_numeric_matrix(
                    T_sym,
                    dim,
                    M,
                    ks_state,
                    Zs_m,
                    Yg_m,
                    **{**freq_p, **cell_p},
                )
                np.testing.assert_allclose(T_grid[f, c], T_ref, atol=1e-12)

    def test_different_cells_give_different_matrices(self, cell_freq_fixture):
        """Cells with different Zs/Yg must produce different matrices."""
        *_, T_grid = cell_freq_fixture
        assert not np.allclose(T_grid[0, 0], T_grid[0, 1])

    def test_different_freqs_give_different_matrices(self, cell_freq_fixture):
        """Different freq points (each with a distinct theta_val) must differ."""
        *_, T_grid = cell_freq_fixture
        # _THETA_FOR_FREQ assigns a unique pump phase per freq point,
        # so off-diagonal coupling entries change between freq slices.
        assert not np.allclose(T_grid[0, 0], T_grid[1, 0])

    def test_freq_axis_independent_of_cell_axis(self, cell_freq_fixture):
        """Changing frequency should not affect the cell-axis ordering."""
        *_, T_grid = cell_freq_fixture
        # cell 0 at freq 0 vs cell 1 at freq 0: same comparison as above
        # but also verify cell 0 at freq 1 vs cell 1 at freq 1 differ
        assert not np.allclose(T_grid[1, 0], T_grid[1, 1])
