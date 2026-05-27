import numpy as np
import pytest

from src.solver.abcd_matrix import ABCDMatrix
from src.solver.s_matrix import SMatrix, ABCD_to_S
from src.symbolic.cell_single_mode import CellSingleMode


def _random_data(rng, Nf, N):
    return rng.standard_normal((Nf, N, N)) + 1j * rng.standard_normal((Nf, N, N))


def _identity_data(Nf, N):
    """ABCD identity = NxN identity matrix (through connection)."""
    eye = np.tile(np.eye(N, dtype=complex), (Nf, 1, 1))
    return eye


class TestProperties:
    def test_Nf_and_N(self):
        rng = np.random.default_rng(10)
        am = ABCDMatrix(_random_data(rng, 7, 6))
        assert am.Nf == 7
        assert am.N == 6

    def test_array_returns_data(self):
        rng = np.random.default_rng(11)
        data = _random_data(rng, 3, 4)
        am = ABCDMatrix(data)
        np.testing.assert_array_equal(am.array, data)

    def test_block_shapes_4x4(self):
        rng = np.random.default_rng(12)
        am = ABCDMatrix(_random_data(rng, 5, 4))
        for block in (am.A, am.B, am.C, am.D):
            assert block.shape == (5, 2, 2)

    def test_block_shapes_6x6(self):
        rng = np.random.default_rng(13)
        am = ABCDMatrix(_random_data(rng, 4, 6))
        for block in (am.A, am.B, am.C, am.D):
            assert block.shape == (4, 3, 3)

    def test_block_values_match_slices(self):
        rng = np.random.default_rng(14)
        data = _random_data(rng, 3, 4)
        am = ABCDMatrix(data)
        np.testing.assert_array_equal(am.A, data[:, :2, :2])
        np.testing.assert_array_equal(am.B, data[:, :2, 2:])
        np.testing.assert_array_equal(am.C, data[:, 2:, :2])
        np.testing.assert_array_equal(am.D, data[:, 2:, 2:])

    def test_repr(self):
        rng = np.random.default_rng(15)
        am = ABCDMatrix(_random_data(rng, 3, 4))
        assert "ABCDMatrix" in repr(am)
        assert "3" in repr(am)
        assert "4" in repr(am)


class TestOperators:
    def test_matmul_shape(self):
        rng = np.random.default_rng(20)
        am1 = ABCDMatrix(_random_data(rng, 5, 4))
        am2 = ABCDMatrix(_random_data(rng, 5, 4))
        assert (am1 @ am2).shape == (5, 4, 4)

    def test_matmul_returns_abcdmatrix(self):
        rng = np.random.default_rng(21)
        am1 = ABCDMatrix(_random_data(rng, 4, 4))
        am2 = ABCDMatrix(_random_data(rng, 4, 4))
        assert isinstance(am1 @ am2, ABCDMatrix)

    def test_matmul_identity_right(self):
        """am @ I == am."""
        rng = np.random.default_rng(22)
        am = ABCDMatrix(_random_data(rng, 5, 4))
        am_id = ABCDMatrix(_identity_data(5, 4))
        np.testing.assert_allclose((am @ am_id).array, am.array, atol=1e-14)

    def test_matmul_identity_left(self):
        """I @ am == am."""
        rng = np.random.default_rng(23)
        am = ABCDMatrix(_random_data(rng, 5, 4))
        am_id = ABCDMatrix(_identity_data(5, 4))
        np.testing.assert_allclose((am_id @ am).array, am.array, atol=1e-14)

    def test_matmul_associativity(self):
        """(A @ B) @ C == A @ (B @ C)."""
        rng = np.random.default_rng(24)
        A = ABCDMatrix(_random_data(rng, 4, 4))
        B = ABCDMatrix(_random_data(rng, 4, 4))
        C = ABCDMatrix(_random_data(rng, 4, 4))
        np.testing.assert_allclose(((A @ B) @ C).array, (A @ (B @ C)).array, atol=1e-12)

    def test_matmul_equals_numpy_matmul(self):
        """ABCDMatrix @ matches batched numpy matmul."""
        rng = np.random.default_rng(25)
        d1 = _random_data(rng, 5, 4)
        d2 = _random_data(rng, 5, 4)
        am1 = ABCDMatrix(d1)
        am2 = ABCDMatrix(d2)
        np.testing.assert_allclose((am1 @ am2).array, d1 @ d2, atol=1e-14)

    def test_matmul_N_mismatch_raises(self):
        rng = np.random.default_rng(26)
        am1 = ABCDMatrix(_random_data(rng, 3, 4))
        am2 = ABCDMatrix(_random_data(rng, 3, 6))
        with pytest.raises(ValueError):
            am1 @ am2

    def test_getitem_returns_abcdmatrix(self):
        rng = np.random.default_rng(27)
        am = ABCDMatrix(_random_data(rng, 6, 4))
        assert isinstance(am[0], ABCDMatrix)

    def test_getitem_single_index_gives_nf1(self):
        rng = np.random.default_rng(28)
        am = ABCDMatrix(_random_data(rng, 6, 4))
        assert am[0].Nf == 1
        assert am[0].N == 4

    def test_getitem_slice(self):
        rng = np.random.default_rng(29)
        am = ABCDMatrix(_random_data(rng, 6, 4))
        assert am[1:4].shape == (3, 4, 4)

    def test_getitem_values(self):
        rng = np.random.default_rng(30)
        data = _random_data(rng, 6, 4)
        am = ABCDMatrix(data)
        np.testing.assert_array_equal(am[2].array[0], data[2])


class TestConversion:
    def test_to_S_returns_smatrix(self):
        rng = np.random.default_rng(40)
        data = _random_data(rng, 3, 4)
        # Make C block well-conditioned
        data[:, 2:, :2] += 2.0 * np.eye(2)
        am = ABCDMatrix(data)
        assert isinstance(am.to_S(Z0=50.0), SMatrix)

    def test_to_S_shape(self):
        rng = np.random.default_rng(41)
        data = _random_data(rng, 5, 4)
        data[:, 2:, :2] += 2.0 * np.eye(2)
        am = ABCDMatrix(data)
        sm = am.to_S(Z0=50.0)
        assert sm.shape == (5, 4, 4)

    def test_to_S_matches_ABCD_to_S(self):
        """to_S agrees with the standalone ABCD_to_S function."""
        rng = np.random.default_rng(42)
        data = _random_data(rng, 4, 4)
        data[:, 2:, :2] += 2.0 * np.eye(2)
        am = ABCDMatrix(data)
        sm = am.to_S(Z0=50.0)
        expected = ABCD_to_S(data, 50.0)
        np.testing.assert_allclose(sm.array, expected, atol=1e-14)

    def test_to_S_z0_propagated(self):
        rng = np.random.default_rng(43)
        data = _random_data(rng, 2, 4)
        data[:, 2:, :2] += 2.0 * np.eye(2)
        am = ABCDMatrix(data)
        assert am.to_S(Z0=75.0).Z0 == 75.0
