"""
Physical consistency checks for the TWPA/TWPC matrices.

1. Manley-Rowe photon conservation
   For a lossless undepleted-pump system the S-matrix maintains the number of photons in-out.

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


def check_manley_rowe_residual(
    S: np.ndarray,
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
) -> np.ndarray:
    """
    Fractional Manley-Rowe residual per frequency per input port.

    Photon conservation requires
        Σᵢ |S[i,j]|² / ωᵢ  =  1 / ωⱼ

    so the returned quantity  (lhs - rhs) / rhs  should be ≈ 0.

    Parameters
    ----------
    S : (Nf, N, N) — complex S-matrix, S[f, output, input]
    omegas : (Nf,) — signal angular frequencies [rad/s]
    omega_pump : pump angular frequency [rad/s]
    ks_state : sideband indices, e.g. [0, 1]

    Returns
    -------
    residual : (Nf, N) — fractional error; 0 = exact conservation
    """
    w = _get_port_frequencies(omegas, omega_pump, ks_state)  # (Nf, N)
    Sabsq = np.abs(S) ** 2  # (Nf, N, N)

    # Sum |S[i,j]|² / ω_i over output ports i (axis -2 = rows)
    weighted_col_sum = (Sabsq / w[:, :, None]).sum(axis=-2)  # (Nf, N)
    reference = 1.0 / w  # (Nf, N)
    return (weighted_col_sum - reference) / reference


def check_power_sum(S: np.ndarray) -> np.ndarray:
    """
    Total output power Σᵢ |S[i,j]|² for each input port j.

    For ε = 0 (passive, lossless): = 1.
    For ε > 0 (active): ≥ 1 at frequencies where the pump transfers energy.

    Returns
    -------
    psum : (Nf, N)
    """
    return (np.abs(S) ** 2).sum(axis=-2)


def check_photon_conservation(
    S: np.ndarray,
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
    ax: plt.Axes | None = None,
) -> np.ndarray:
    """
    Generalized Manley-Rowe photon conservation check.

    Computes  Σᵢ (ωⱼ/ωᵢ) |S[i,j]|²  for every input port j.
    A lossless system returns 1 for all ports and frequencies.

    Parameters
    ----------
    S        : (Nf, N, N) — complex S-matrix, S[f, output, input]
    omegas   : (Nf,) — signal angular frequencies [rad/s]
    omega_pump : pump angular frequency [rad/s]
    ks_state : sideband indices, e.g. [0, 1]
    ax       : optional Axes; a new figure is created if None

    Returns
    -------
    check : (Nf, N)
    """
    w = _get_port_frequencies(omegas, omega_pump, ks_state)  # (Nf, N)
    Sabsq = np.abs(S) ** 2  # (Nf, N, N);  S[f, output_i, input_j]

    # Broadcast port frequencies into the (Nf, output_i, input_j) shape
    omega_j = w[:, np.newaxis, :]  # (Nf,  1,  N) — freq of input port j
    omega_i = w[:, :, np.newaxis]  # (Nf,  N,  1) — freq of output port i

    # check[f, j] = Σᵢ (ω_j / ω_i) |S[f, i, j]|²  —  should equal 1 when lossless
    check = (Sabsq * (omega_j / omega_i)).sum(axis=1)  # sum over output ports → (Nf, N)
    # check = Sabsq[:, 0, 0] + omegas/(omegas+omega_pump) * Sabsq[:, 1, 0]+Sabsq[:, 2, 0] + omegas/(omegas+omega_pump) * Sabsq[:, 3, 0]
    if ax is None:
        _, ax = plt.subplots()

    n_modes = len(ks_state)
    freqs_ghz = omegas / (2 * np.pi)
    for j in range(check.shape[1]):
        side = "L" if j < n_modes else "R"
        k = ks_state[j % n_modes]
        ax.plot(freqs_ghz, check[:, j], label=f"port {j + 1} (k={k}, {side})")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"$\sum_i (\omega_j/\omega_i)\,|S_{ij}|^2$")
    ax.set_title("Photon conservation (Manley-Rowe)")
    ax.legend()

    return check

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

