import pytest

from backends import Backend, available_backends, get_backend, register_backend
from backends.numpy_backend import NumpyBackend


class TestGetBackend:
    def test_default_is_numpy(self):
        assert isinstance(get_backend(), NumpyBackend)

    def test_explicit_name(self):
        assert isinstance(get_backend("numpy"), NumpyBackend)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("does-not-exist")

    def test_returns_cached_instance(self):
        assert get_backend("numpy") is get_backend("numpy")

    def test_env_var_selects_backend(self, monkeypatch):
        monkeypatch.setenv("TWPA_BACKEND", "numpy")
        assert isinstance(get_backend(), NumpyBackend)

    def test_explicit_name_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("TWPA_BACKEND", "does-not-exist")
        assert isinstance(get_backend("numpy"), NumpyBackend)


class TestRegisterBackend:
    def test_register_and_resolve(self):
        register_backend("numpy-alias", "backends.numpy_backend", "NumpyBackend")
        assert isinstance(get_backend("numpy-alias"), NumpyBackend)
        assert "numpy-alias" in available_backends()

    def test_register_unimportable_module_raises_on_use(self):
        register_backend("broken", "backends.does_not_exist", "Nope")
        with pytest.raises(ImportError):
            get_backend("broken")

    def test_register_missing_class_raises_on_use(self):
        register_backend("broken-class", "backends.numpy_backend", "NoSuchBackend")
        with pytest.raises(AttributeError):
            get_backend("broken-class")


class TestAvailableBackends:
    def test_includes_numpy(self):
        assert "numpy" in available_backends()

    def test_resolved_instance_is_a_backend(self):
        assert isinstance(get_backend("numpy"), Backend)
