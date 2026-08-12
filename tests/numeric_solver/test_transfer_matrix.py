import numpy as np
import pytest

from src.numerical_solver.tranfer_matrix import (
    single_mode_matrix,
    single_mode_matrix_grid,
    symmetric_single_mode_matrix,
)
from src.models.cell import CellImmitance


def _random_zy(rng, Nf, m):
    Zs = rng.standard_normal((Nf, m, m)) + 1j * rng.standard_normal((Nf, m, m))
    Yg = rng.standard_normal((Nf, m, m)) + 1j * rng.standard_normal((Nf, m, m))
    return Zs, Yg


def _make_cell(Zs, Yg):
    return CellImmitance(
        theta=0.0,
        Zs0_fn=lambda w: 0.0,
        Yg0_fn=lambda w: 0.0,
        Zs_harm_fn=Zs,
        Yg_harm_fn=Yg,
    )


class TestSingleModeMatrix:
    def test_known_1x1_result(self):
        """m=1 scalar case: T = [[1, -Zs], [-Yg, 1 + Yg*Zs]]."""
        Zs = np.array([[[2.0]]], dtype=complex)
        Yg = np.array([[[3.0]]], dtype=complex)
        expected = np.array([[[1.0, -2.0], [-3.0, 1.0 + 3.0 * 2.0]]], dtype=complex)
        np.testing.assert_allclose(single_mode_matrix(Zs, Yg), expected)

    def test_shape(self):
        rng = np.random.default_rng(0)
        Zs, Yg = _random_zy(rng, 5, 4)
        assert single_mode_matrix(Zs, Yg).shape == (5, 8, 8)

    def test_zero_coupling_is_block_diagonal_identity_plus(self):
        """Zs=Yg=0 reduces to the identity ABCD (a through connection)."""
        Nf, m = 3, 2
        Zs = np.zeros((Nf, m, m), dtype=complex)
        Yg = np.zeros((Nf, m, m), dtype=complex)
        T = single_mode_matrix(Zs, Yg)
        expected = np.broadcast_to(np.eye(2 * m, dtype=complex), (Nf, 2 * m, 2 * m))
        np.testing.assert_allclose(T, expected)


class TestSymmetricSingleModeMatrix:
    def test_shape(self):
        rng = np.random.default_rng(1)
        Zs, Yg = _random_zy(rng, 5, 4)
        assert symmetric_single_mode_matrix(Zs, Yg).shape == (5, 8, 8)

    def test_zero_coupling_is_identity(self):
        Nf, m = 3, 2
        Zs = np.zeros((Nf, m, m), dtype=complex)
        Yg = np.zeros((Nf, m, m), dtype=complex)
        T = symmetric_single_mode_matrix(Zs, Yg)
        expected = np.broadcast_to(np.eye(2 * m, dtype=complex), (Nf, 2 * m, 2 * m))
        np.testing.assert_allclose(T, expected)

    def test_differs_from_L_topology_for_nonzero_coupling(self):
        rng = np.random.default_rng(2)
        Zs, Yg = _random_zy(rng, 2, 2)
        L = single_mode_matrix(Zs, Yg)
        pi = symmetric_single_mode_matrix(Zs, Yg)
        assert not np.allclose(L, pi)


class TestSingleModeMatrixGrid:
    @pytest.mark.parametrize("topology", ["L", "pi"])
    def test_matches_per_cell_call(self, topology):
        rng = np.random.default_rng(3)
        Nf, m, Ncells = 4, 3, 5
        cells = [_make_cell(*_random_zy(rng, Nf, m)) for _ in range(Ncells)]

        T_grid = single_mode_matrix_grid(cells, topology)
        assert T_grid.shape == (Nf, Ncells, 2 * m, 2 * m)

        fn = single_mode_matrix if topology == "L" else symmetric_single_mode_matrix
        for c, cell in enumerate(cells):
            np.testing.assert_allclose(
                T_grid[:, c], fn(cell.Zs_harm_fn, cell.Yg_harm_fn), atol=1e-12
            )


class TestBackendParam:
    """`backend` accepts an instance or a name string and matches the default path."""

    def test_single_mode_matrix_name_string_matches_default(self):
        rng = np.random.default_rng(4)
        Zs, Yg = _random_zy(rng, 3, 2)
        np.testing.assert_allclose(
            single_mode_matrix(Zs, Yg),
            single_mode_matrix(Zs, Yg, backend="numpy"),
            atol=1e-14,
        )

    def test_symmetric_single_mode_matrix_name_string_matches_default(self):
        rng = np.random.default_rng(5)
        Zs, Yg = _random_zy(rng, 3, 2)
        np.testing.assert_allclose(
            symmetric_single_mode_matrix(Zs, Yg),
            symmetric_single_mode_matrix(Zs, Yg, backend="numpy"),
            atol=1e-14,
        )

    def test_unknown_backend_name_raises(self):
        rng = np.random.default_rng(6)
        Zs, Yg = _random_zy(rng, 3, 2)
        with pytest.raises(ValueError, match="Unknown backend"):
            single_mode_matrix(Zs, Yg, backend="does-not-exist")
