"""
Dispersion relation extraction from transfer matrices.
"""

from __future__ import annotations
import numpy as np


def bloch_wavenumbers(
    T_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bloch propagation constants for an NxN unit cell via eigenvalue decomposition.

    For N = 2 this is equivalent to two_port_dispersion().
    For N > 2 (coupled sidebands from pump modulation), all N Bloch-mode
    wavenumbers are returned simultaneously.


    Parameters
    ----------
    T_cell : ABCDMatrix, shape (Nf, N, N)
        Forward transfer matrix of ONE unit cell.

    Returns
    -------
    alpha : ndarray, shape (Nf, N)
        Attenuation per cell (real part of γ_i), sorted descending by k.
    k : ndarray, shape (Nf, N)
        Wavenumber rad/cell (imaginary part of γ_i), sorted descending.
    """
    eigenvalues = np.linalg.eigvals(T_cell)  # (Nf, N)
    gamma = -np.log(eigenvalues)  # γ_i = α_i + j·k_i
    idx = np.argsort(gamma.imag, axis=-1)[:, ::-1]  # descending k
    return (
        np.take_along_axis(gamma.real, idx, axis=-1),
        np.take_along_axis(gamma.imag, idx, axis=-1),
    )
