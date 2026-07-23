import numpy as np

from backends import Backend, get_backend


def _resolve_backend(backend: Backend | str | None) -> Backend:
    return backend if isinstance(backend, Backend) else get_backend(backend)


def single_mode_matrix(Zs, Yg, backend: Backend | str | None = None) -> np.ndarray:
    """
    Per-cell ABCD matrix for the "L" (series-then-shunt) topology.

    Parameters
    ----------
    Zs, Yg : ndarray, shape (Nf, m, m)
        Series impedance and shunt admittance harmonic matrices.
    backend : Backend, backend name, or None
        Numerical backend to run the kernel on. Defaults to "numpy", or
        the $TWPA_BACKEND environment variable if set. See
        `backends.available_backends()`.

    Returns
    -------
    ndarray, shape (Nf, 2m, 2m)
    """
    return _resolve_backend(backend).single_mode_matrix(
        np.asarray(Zs, dtype=complex), np.asarray(Yg, dtype=complex)
    )


def symmetric_single_mode_matrix(Zs, Yg, backend: Backend | str | None = None) -> np.ndarray:
    """
    Per-cell ABCD matrix for the symmetric "pi" (shunt/2-series-shunt/2) topology.

    Parameters
    ----------
    Zs, Yg : ndarray, shape (Nf, m, m)
        Series impedance and shunt admittance harmonic matrices.
    backend : Backend, backend name, or None
        Numerical backend to run the kernel on. Defaults to "numpy", or
        the $TWPA_BACKEND environment variable if set. See
        `backends.available_backends()`.

    Returns
    -------
    ndarray, shape (Nf, 2m, 2m)
    """
    return _resolve_backend(backend).symmetric_single_mode_matrix(
        np.asarray(Zs, dtype=complex), np.asarray(Yg, dtype=complex)
    )


def single_mode_matrix_grid(cells, topology, backend: Backend | str | None = None) -> np.ndarray:
    """
    Build the per-cell, per-frequency ABCD matrix grid for a whole cascade.

    Parameters
    ----------
    cells : list of CellImmitance
        Each cell's `Zs_harm_fn` / `Yg_harm_fn` must already be arrays of
        shape (Nf, m, m), not callables.
    topology : "L" or "pi"
    backend : Backend, backend name, or None
        Numerical backend to run the kernel on. Defaults to "numpy", or
        the $TWPA_BACKEND environment variable if set. Backends that fuse
        the whole cell loop into one compiled kernel (e.g. "numba") avoid
        Ncells separate per-cell dispatches -- see
        `Backend.single_mode_matrix_grid`.

    Returns
    -------
    ndarray, shape (Nf, Ncells, 2m, 2m)
    """
    return _resolve_backend(backend).single_mode_matrix_grid(cells, topology)
