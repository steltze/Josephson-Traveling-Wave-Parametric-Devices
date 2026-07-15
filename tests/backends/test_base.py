import numpy as np
import pytest

from backends.base import Backend
from backends.numpy_backend import NumpyBackend


class IncompleteBackend(Backend):
    """Missing `redheffer_star` — should not be instantiable."""

    name = "incomplete"

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
        s_cells = rng.standard_normal((3, 1, 4, 4)) + 1j * rng.standard_normal((3, 1, 4, 4))
        backend = NumpyBackend()
        np.testing.assert_array_equal(backend.cascade_all(s_cells), s_cells[:, 0])
