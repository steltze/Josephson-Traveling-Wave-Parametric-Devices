"""
Physical consistency checks for the TWPA/TWPC matrices.

1. Photon-flux conservation

2. Transfer matrix determinant (for a lossless material) should be 1.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from logger import get_logger

log = get_logger(__name__)


def _get_port_frequencies(
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
) -> np.ndarray:
    """
    Angular frequency assigned to each S-matrix port.

    Port ordering mirrors the Floquet state vector:
        [mode_k0_L, mode_k1_L, …, mode_k0_R, mode_k1_R, …]

    Returns
    -------
    port_omegas : ndarray, shape (Nf, N)
    """
    n_modes = len(ks_state)
    N = 2 * n_modes
    port_omegas = np.empty((len(omegas), N))
    for i, k in enumerate(ks_state):
        freq = omegas + k * omega_pump
        port_omegas[:, i] = freq  # left port
        port_omegas[:, n_modes + i] = freq  # right port
    return port_omegas


def check_photon_flux_conservation(
    S_ph: np.ndarray,
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
) -> np.ndarray:
    """
    eta-weighted power sum for a photon-flux-normalized S-matrix.

    ``S_ph`` must come from photon-flux normalization (e.g.
    ``Simulation.get_s_matrix(normalize=True)`` or
    ``SMatrix.normalize_photon_flux``). A port's signed frequency
    ωₖ = ω + k·ω_pump can be negative (an idler / down-converted sideband,
    common whenever ks_state contains a negative k) — such ports are
    pseudo-unitary partners of the positive-frequency ports, not ordinary
    channels. A *plain* Σᵢ|S_ph[i,j]|² is not conserved for them and can
    look wildly "overestimated" wherever that port carries real parametric
    gain. The conserved quantity (Manley-Rowe photon number, exact for a
    lossless-junction line even with gain) is instead:

        Σᵢ ηᵢ |S_ph[i,j]|²  =  ηⱼ,      ηₖ = sign(ωₖ)

    Returns
    -------
    check : (Nf, N) — should equal ηⱼ (±1) for every input port j; deviation
        indicates real dissipation or a port-labeling/normalization bug.
    """
    w = _get_port_frequencies(omegas, omega_pump, ks_state)  # (Nf, N) signed
    eta = np.sign(w)
    Sabsq = np.abs(S_ph) ** 2  # (Nf, N, N)
    return (Sabsq * eta[:, :, np.newaxis]).sum(axis=1)  # sum over output i


def check_transfer_matrix_determinant(
    T_grid: np.ndarray,  # (Nf, Nc, N, N)
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the (Nf, Nc) indices where |det(T) - 1| >= tolerance.
    """
    dets = np.abs(np.linalg.det(T_grid))  # (Nf, Nc)
    violating = np.abs(dets - 1.0) >= tolerance
    nf_idx, nc_idx = np.where(violating)
    if nf_idx.size:
        log.error(f"{nf_idx.size} cells with |det(T)-1| >= {tolerance}.")
    else:
        log.test("Determinant check pass!")
    return nf_idx, nc_idx
