from __future__ import annotations

import importlib
import os

from backends.base import Backend

_ENV_VAR = "TWPA_BACKEND"

# name -> (module path, class name). Registered lazily: the module is only
# imported when that backend is actually requested, so an accelerator with
# an optional dependency (numba, JAX, ...) doesn't force that dependency on
# everyone who imports this package.
_REGISTRY: dict[str, tuple[str, str]] = {
    "numpy": ("backends.numpy_backend", "NumpyBackend"),
    "numba": ("backends.numba_backend", "NumbaBackend"),
}

_instances: dict[str, Backend] = {}


def register_backend(name: str, module: str, cls_name: str) -> None:
    """
    Register a Backend implementation under `name`.

    Parameters
    ----------
    name : str
        Key used to select this backend from `get_backend`.
    module : str
        Dotted import path of the module defining the backend, e.g.
        "backends.numba_backend".
    cls_name : str
        Name of the `Backend` subclass within that module.
    """
    _REGISTRY[name] = (module, cls_name)
    _instances.pop(name, None)


def available_backends() -> list[str]:
    """Names of all registered backends (registration does not guarantee importability)."""
    return sorted(_REGISTRY)


def get_backend(name: str | None = None) -> Backend:
    """
    Resolve a backend instance by name.

    Resolution order: explicit `name` -> `$TWPA_BACKEND` env var -> "numpy".
    Instances are cached, so repeated calls with the same name return the
    same object.

    Raises
    ------
    ValueError
        `name` is not a registered backend.
    ImportError
        The backend's module could not be imported (e.g. a missing
        optional dependency).
    """
    if name is None:
        name = os.environ.get(_ENV_VAR, "numpy")
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {name!r}. Available backends: {available_backends()}"
        )
    if name not in _instances:
        module_path, cls_name = _REGISTRY[name]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Backend {name!r} could not be imported ({exc}). "
                "It may need an optional dependency installed."
            ) from exc
        _instances[name] = getattr(module, cls_name)()
    return _instances[name]
