"""
Dispersion relation extraction from transfer matrices.
"""

from __future__ import annotations

import numpy as np

from solver.abcd_matrix import ABCDMatrix


def bloch_wavenumbers(
    T_cell: ABCDMatrix,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bloch propagation constants for an NxN unit cell via eigenvalue decomposition.

    Each eigenvalue λ_i of the forward transfer matrix T satisfies:
        state(n+1) = λ_i · state(n)   (Bloch condition)
    giving propagation constant γ_i = −log(λ_i) = α_i + j·k_i.

    For N = 2 this is equivalent to two_port_dispersion().
    For N > 2 (coupled sidebands from pump modulation), all N Bloch-mode
    wavenumbers are returned simultaneously.

    The N modes come in ±k pairs for a lossless reciprocal ATL:
    forward-propagating modes have k > 0, backward-propagating k < 0.
    In passbands |α| ≈ 0; in stopbands |α| > 0 and k = 0 or π.

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
    eigenvalues = np.linalg.eigvals(T_cell.array)  # (Nf, N)
    gamma = -np.log(eigenvalues)                    # γ_i = α_i + j·k_i
    idx = np.argsort(gamma.imag, axis=-1)[:, ::-1]  # descending k
    return (
        np.take_along_axis(gamma.real, idx, axis=-1),
        np.take_along_axis(gamma.imag, idx, axis=-1),
    )
