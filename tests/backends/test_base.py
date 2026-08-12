import numpy as np
import pytest

from backends.base import Backend
from backends.numpy_backend import NumpyBackend


class IncompleteBackend(Backend):
    """Missing `redheffer_star` — should not be instantiable."""

    name = "incomplete"

    def single_mode_matrix(self, Zs, Yg):
        return Zs

    def symmetric_single_mode_matrix(self, Zs, Yg):
        return Zs

    def abcd_to_s(self, abcd, z0):
        return abcd


class TestBackendIsAbstract:
    def test_cannot_instantiate_backend_directly(self):
        with pytest.raises(TypeError):
            Backend()

    def test_cannot_instantiate_incomplete_subclass(self):
        with pytest.raises(TypeError):
            IncompleteBackend()


class TestDefaultCascadeAll:
    """`Backend.cascade_all`'s default reduce implementation, exercised via NumpyBackend."""

    def test_matches_manual_reduce(self):
        rng = np.random.default_rng(0)
        Nf, Nc, N = 5, 6, 4
        s_cells = rng.standard_normal((Nf, Nc, N, N)) + 1j * rng.standard_normal(
            (Nf, Nc, N, N)
        )
        backend = NumpyBackend()

        expected = s_cells[:, 0]
        for c in range(1, Nc):
            expected = backend.redheffer_star(s_cells[:, c], expected)

        np.testing.assert_allclose(backend.cascade_all(s_cells), expected, atol=1e-12)

    def test_single_cell_is_identity(self):
        rng = np.random.default_rng(1)
        s_cells = rng.standard_normal((3, 1, 4, 4)) + 1j * rng.standard_normal(
            (3, 1, 4, 4)
        )
        backend = NumpyBackend()
        np.testing.assert_array_equal(backend.cascade_all(s_cells), s_cells[:, 0])


class _StubCell:
    def __init__(self, Zs, Yg):
        self.Zs_harm_fn = Zs
        self.Yg_harm_fn = Yg


class TestDefaultSingleModeMatrixGrid:
    """`Backend.single_mode_matrix_grid`'s default per-cell loop, exercised via NumpyBackend."""

    @pytest.mark.parametrize("topology", ["L", "pi"])
    def test_matches_per_cell_calls(self, topology):
        rng = np.random.default_rng(2)
        Nf, m, Ncells = 3, 2, 4
        cells = [
            _StubCell(
                rng.standard_normal((Nf, m, m)) + 1j * rng.standard_normal((Nf, m, m)),
                rng.standard_normal((Nf, m, m)) + 1j * rng.standard_normal((Nf, m, m)),
            )
            for _ in range(Ncells)
        ]
        backend = NumpyBackend()
        fn = (
            backend.single_mode_matrix
            if topology == "L"
            else backend.symmetric_single_mode_matrix
        )

        T_grid = backend.single_mode_matrix_grid(cells, topology)
        assert T_grid.shape == (Nf, Ncells, 2 * m, 2 * m)
        for c, cell in enumerate(cells):
            np.testing.assert_allclose(
                T_grid[:, c], fn(cell.Zs_harm_fn, cell.Yg_harm_fn)
            )
