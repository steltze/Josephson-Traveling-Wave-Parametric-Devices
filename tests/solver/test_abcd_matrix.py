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


class TestFromCellGrid:
    def test_shape(self):
        rng = np.random.default_rng(50)
        data = _random_data(rng, 5, 4)[None].repeat(3, axis=0)  # (5,4,4) -> (5,3,4,4)
        data = rng.standard_normal((7, 4, 4, 4)) + 1j * rng.standard_normal(
            (7, 4, 4, 4)
        )
        am = ABCDMatrix.from_cell_grid(data)
        assert am.shape == (7, 4, 4)

    def test_returns_abcd_matrix(self):
        rng = np.random.default_rng(51)
        data = rng.standard_normal((5, 3, 4, 4)) + 1j * rng.standard_normal(
            (5, 3, 4, 4)
        )
        assert isinstance(ABCDMatrix.from_cell_grid(data), ABCDMatrix)

    def test_single_cell_equals_direct_construction(self):
        """With Nc=1 the result must equal ABCDMatrix(data[:,0])."""
        rng = np.random.default_rng(52)
        data = rng.standard_normal((5, 1, 4, 4)) + 1j * rng.standard_normal(
            (5, 1, 4, 4)
        )
        am = ABCDMatrix.from_cell_grid(data)
        np.testing.assert_allclose(am.array, data[:, 0], atol=1e-14)

    def test_two_cells_equals_matmul(self):
        """from_cell_grid with Nc=2 must equal data[:,0] @ data[:,1]."""
        rng = np.random.default_rng(53)
        data = rng.standard_normal((6, 2, 4, 4)) + 1j * rng.standard_normal(
            (6, 2, 4, 4)
        )
        am = ABCDMatrix.from_cell_grid(data)
        expected = data[:, 0] @ data[:, 1]
        np.testing.assert_allclose(am.array, expected, atol=1e-14)

    def test_three_cells_equals_sequential_matmul(self):
        """Nc=3: result == data[:,0] @ data[:,1] @ data[:,2]."""
        rng = np.random.default_rng(54)
        data = rng.standard_normal((4, 3, 6, 6)) + 1j * rng.standard_normal(
            (4, 3, 6, 6)
        )
        am = ABCDMatrix.from_cell_grid(data)
        expected = data[:, 0] @ data[:, 1] @ data[:, 2]
        np.testing.assert_allclose(am.array, expected, atol=1e-12)

    def test_identity_cells_give_identity(self):
        """Cascading Nc identity matrices must return the identity."""
        Nf, Nc, N = 5, 4, 4
        eye = np.tile(np.eye(N, dtype=complex), (Nf, Nc, 1, 1))
        am = ABCDMatrix.from_cell_grid(eye)
        np.testing.assert_allclose(am.array, np.tile(np.eye(N), (Nf, 1, 1)), atol=1e-14)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            ABCDMatrix.from_cell_grid(np.zeros((3, 4, 4)))  # ndim=3, not 4

    def test_to_s_pipeline(self):
        """from_cell_grid(...).to_S(...) must produce a valid SMatrix."""
        rng = np.random.default_rng(55)
        data = rng.standard_normal((5, 3, 4, 4)) + 1j * rng.standard_normal(
            (5, 3, 4, 4)
        )
        # Make C blocks well-conditioned so ABCD→S does not blow up
        data[:, :, 2:, :2] += 2.0 * np.eye(2)
        sm = ABCDMatrix.from_cell_grid(data).to_S(Z0=50.0)
        assert isinstance(sm, SMatrix)
        assert sm.shape == (5, 4, 4)
