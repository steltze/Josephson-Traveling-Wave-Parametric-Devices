import pytest

from backends.base import Backend


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
