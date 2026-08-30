"""
Dispersion relation extraction from transfer matrices.
"""

from __future__ import annotations

import numpy as np


def bloch_wavenumbers(
    T_cell: np.ndarray,
    return_eigenvectors: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bloch propagation constants for an NxN unit cell via eigenvalue decomposition.

    For N = 2 this is equivalent to two_port_dispersion().
    For N > 2 (coupled sidebands from pump modulation), all N Bloch-mode
    wavenumbers are returned simultaneously.


    Parameters
    ----------
    T_cell : ABCDMatrix, shape (Nf, N, N)
        Forward transfer matrix of ONE unit cell.
    return_eigenvectors : bool
        If True, also return the eigenvectors (reordered to match alpha/k),
        e.g. to classify which sideband dominates each Bloch mode.

    Returns
    -------
    alpha : ndarray, shape (Nf, N)
        Attenuation per cell (real part of γ_i), sorted descending by k.
    k : ndarray, shape (Nf, N)
        Wavenumber rad/cell (imaginary part of γ_i), sorted descending.
    eigenvectors : ndarray, shape (Nf, N, N), only if return_eigenvectors
        Column i is the eigenvector for mode i (alpha[:, i], k[:, i]).
    """
    if return_eigenvectors:
        eigenvalues, eigenvectors = np.linalg.eig(T_cell)  # (Nf, N), (Nf, N, N)
    else:
        eigenvalues = np.linalg.eigvals(T_cell)  # (Nf, N)
    gamma = -np.log(eigenvalues)  # γ_i = α_i + j·k_i
    idx = np.argsort(gamma.imag, axis=-1)[:, ::-1]  # descending k
    alpha = np.take_along_axis(gamma.real, idx, axis=-1)
    k = np.take_along_axis(gamma.imag, idx, axis=-1)
    if return_eigenvectors:
        eigenvectors = np.take_along_axis(eigenvectors, idx[:, None, :], axis=-1)
        return alpha, k, eigenvectors
    return alpha, k
