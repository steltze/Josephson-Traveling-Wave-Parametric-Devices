"""
Physical consistency checks for the TWPA/TWPC matrices.

1. Manley-Rowe photon conservation

2. Energy conservation check


3. Pump photon balance (3-wave formula)

4. Transfer matrix determinant (for a lossless material) should be 1.
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

def _get_port_sideband_indices(ks_state: list[int]) -> np.ndarray:
    """
    Sideband index k assigned to each S-matrix port.

    Port ordering: [mode_k0_L, mode_k1_L, …, mode_k0_R, mode_k1_R, …]

    Returns
    -------
    k_ports : ndarray, shape (N,)  — integer sideband index per port
    """
    n_modes = len(ks_state)
    k_ports = np.empty(2 * n_modes, dtype=int)
    for i, k in enumerate(ks_state):
        k_ports[i] = k
        k_ports[n_modes + i] = k
    return k_ports


def check_energy_conservation(
    S: np.ndarray,
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
    ax: plt.Axes | None = None,
) -> np.ndarray:
    """
    Energy conservation check per input port.

    Computes  Σᵢ (ωᵢ/ωⱼ) |S[i,j]|²  for every input port j.

    For a passive lossless system this equals 1.
    The deviation from 1 is the net pump energy drawn:
        ΔE_pump(j) = (check(j) − 1) · ωⱼ

    Positive deviation: pump provides energy (gain region).
    Negative deviation: pump absorbs energy (deamplification / gap).

    Returns
    -------
    check : (Nf, N)  — 1 = energy conserved, deviation = pump exchange
    """
    w = _get_port_frequencies(omegas, omega_pump, ks_state)  # (Nf, N)
    Sabsq = np.abs(S) ** 2  # (Nf, N, N)

    omega_i = w[:, :, np.newaxis]  # (Nf, N, 1) — freq of output port i
    omega_j = w[:, np.newaxis, :]  # (Nf, 1, N) — freq of input port j

    check = (Sabsq * (omega_i / omega_j)).sum(axis=1)  # sum over output i → (Nf, N)

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
    ax.set_ylabel(r"$\sum_i (\omega_i/\omega_j)\,|S_{ij}|^2$")
    ax.set_title("Energy conservation  (deviation = pump energy exchange)")
    ax.legend()

    return check


def check_pump_photon_balance(
    S: np.ndarray,
    omegas: np.ndarray,
    omega_pump: float,
    ks_state: list[int],
    ax: plt.Axes | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pump energy balance using the 3-wave sideband formula.

    For the process  ωₖ = ωₛ + k·ωₚ, each transition from sideband kⱼ to kᵢ
    consumes (kᵢ − kⱼ) pump photons.  The implied pump energy drawn per unit
    input at port j is:

        ΔE_pump_3wave(j) = ωₚ · Σᵢ (kᵢ − kⱼ) · |S_ij|² / ωᵢ

    The actual pump energy drawn from the energy balance is:

        ΔE_pump_actual(j) = Σᵢ ωᵢ|S_ij|² − ωⱼ

    For a lossless single-step 3-wave system these are equal (residual ≈ 0).
    For many cells near the gap the residual is nonzero because the evanescent
    standing wave undergoes multi-step pump exchange not captured by counting
    only the visible output photons.

    Parameters
    ----------
    S          : (Nf, N, N) — complex S-matrix, S[f, output, input]
    omegas     : (Nf,) — signal angular frequencies
    omega_pump : pump angular frequency
    ks_state   : sideband indices, e.g. [0, 1]

    Returns
    -------
    dE_actual  : (Nf, N)  — pump energy drawn from energy balance
    dE_3wave   : (Nf, N)  — pump energy from 3-wave sideband counting
    residual   : (Nf, N)  — dE_actual - dE_3wave  (0 = pure single-step)
    """
    w = _get_port_frequencies(omegas, omega_pump, ks_state)  # (Nf, N)
    k_ports = _get_port_sideband_indices(ks_state)  # (N,)
    Sabsq = np.abs(S) ** 2  # (Nf, N, N)

    omega_i = w[:, :, np.newaxis]  # (Nf, N, 1)
    omega_j = w[:, np.newaxis, :]  # (Nf, 1, N)

    # Actual pump energy drawn: E_out − E_in
    dE_actual = (Sabsq * omega_i).sum(axis=1) - w  # (Nf, N)

    # 3-wave formula: ωₚ · Σᵢ (kᵢ−kⱼ) · |S_ij|² / ωᵢ
    delta_k = k_ports[:, np.newaxis] - k_ports[np.newaxis, :]  # (N, N): delta_k[i,j] = kᵢ−kⱼ
    dE_3wave = omega_pump * (Sabsq * delta_k[np.newaxis] / omega_i).sum(axis=1)  # (Nf, N)

    residual = dE_actual - dE_3wave  # (Nf, N)

    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
    else:
        axes = [ax, ax]

    n_modes = len(ks_state)
    freqs_ghz = omegas / (2 * np.pi)

    ax0, ax1 = axes
    for j in range(dE_actual.shape[1]):
        side = "L" if j < n_modes else "R"
        k = ks_state[j % n_modes]
        lbl = f"port {j + 1} (k={k}, {side})"
        ax0.plot(freqs_ghz, dE_actual[:, j] / omega_pump, label=lbl)
        ax1.plot(freqs_ghz, residual[:, j] / omega_pump, label=lbl)

    for a, title, ylabel in [
        (ax0, "Pump energy drawn  (units of ωₚ per input photon)", "ΔE_pump / ωₚ"),
        (ax1, "Residual: actual − 3wave  (0 = pure single-step)", "residual / ωₚ"),
    ]:
        a.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        a.set_xlabel("Frequency (GHz)")
        a.set_ylabel(ylabel)
        a.set_title(title)
        a.legend(fontsize=8)

    return dE_actual, dE_3wave, residual


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

