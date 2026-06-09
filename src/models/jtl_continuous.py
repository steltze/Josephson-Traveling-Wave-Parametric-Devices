"""
Continuous-limit backward-wave coupled-mode model for the JTL TWPA/TWPC.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

def dispersion_bloch_with_plasma(
    omegas: np.ndarray, omega_cutoff: float, omega_j: float
) -> np.ndarray:
    """Bloch wavenumber k(ω) in rad/cell — exact discrete JTL dispersion.

    k·a = arccos(1 - 2(ω/ωc)² / (1 - ω²/ωj²))
    """
    w = np.asarray(omegas, dtype=float)
    denom = 1.0 - (w / omega_j) ** 2
    arg = np.clip(1.0 - 2.0 * (w / omega_cutoff) ** 2 / denom, -1.0, 1.0)
    return np.arccos(arg)

def dispersion_linear(
    omegas: np.ndarray, omega_cutoff: float
) -> np.ndarray:
    """Wavenumber k(ω) in rad/cell — pure linear dispersion, no JJ.

    k·a = 2ω/ωc
    """
    return 2.0 * np.asarray(omegas, dtype=float) / omega_cutoff

def dispersion_bloch(
    omegas: np.ndarray, omega_cutoff: float
) -> np.ndarray:
    """Bloch wavenumber k(ω) in rad/cell — exact discrete LC ladder, no JJ.

    k·a = 2·arcsin(ω/ωc)
    """
    return 2.0 * np.arcsin(np.clip(np.asarray(omegas, dtype=float) / omega_cutoff, 0.0, 1.0))


def dispersion_linear_with_plasma(
    omegas: np.ndarray, omega_cutoff: float, omega_j: float
) -> np.ndarray:
    """Wavenumber k(ω) in rad/cell — continuous approximation with JJ correction.

    k·a = 2ω/ωc / √(1 - ω²/ωj²)
    """
    w = np.asarray(omegas, dtype=float)
    return 2.0 * w / (omega_cutoff * np.sqrt(1.0 - (w / omega_j) ** 2))

def _dk_dw(omega: float, omega_cutoff: float, omega_j: float) -> float:
    """
    d(k·a)/dω for the continuous JTL dispersion.\
    """
    return 2.0 / (omega_cutoff * (1.0 - (omega / omega_j) ** 2))


def phase_mismatch(
    omegas: np.ndarray,
    omega_pump: float,
    omega_cutoff: float,
    v_pump: float,
    cell_size: float,
    omega_j: float = np.inf,
) -> np.ndarray:
    """Backward-wave phase mismatch Δk = ks + kI - kp  in rad/cell.

    Zero at the gap-centre frequency, positive above it, negative below.
    """
    omegas = np.asarray(omegas, dtype=float)
    omega_I = omegas + omega_pump
    kp = omega_pump * cell_size / v_pump  # rad/cell
    ks = dispersion_linear_with_plasma(omegas, omega_cutoff, omega_j)
    kI = dispersion_linear_with_plasma(omega_I, omega_cutoff, omega_j)
    return ks + kI - kp


def gap_center(
    omega_pump: float,
    omega_cutoff: float,
    v_pump: float,
    cell_size: float,
    omega_j: float = np.inf,
) -> float | None:
    """Solve ks(ωgap) + kI(ωgap + ωp) = kp for ωgap.

    Returns None if no solution exists.
    """
    kp = omega_pump * cell_size / v_pump

    def residual(ws: float) -> float:
        omega_I = ws + omega_pump
        if ws <= 0.0:
            return float("nan")
        return dispersion_linear_with_plasma(ws, omega_cutoff, omega_j) + dispersion_linear_with_plasma(omega_I, omega_cutoff, omega_j) - kp

    lo = omega_cutoff * 1e-4
    hi = (omega_cutoff - omega_pump) * 0.999
    if hi <= lo:
        return None
    r_lo, r_hi = residual(lo), residual(hi)
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)):
        return None
    if r_lo * r_hi > 0:
        return None
    return brentq(residual, lo, hi)


def gap_bandwidth(
    omega_pump: float,
    omega_cutoff: float,
    omega_j: float,
    epsilon: float,
    v_pump: float,
    cell_size: float,
) -> tuple[float | None, float | None]:
    """
    Analytical stop-band bandwidth from the condition.
    """
    omega_gap = gap_center(omega_pump, omega_cutoff, v_pump, cell_size, omega_j)
    if omega_gap is None:
        return None, None

    omega_I_gap = omega_gap + omega_pump
    kS0 = dispersion_linear_with_plasma(omega_gap, omega_cutoff, omega_j)
    kI0 = dispersion_linear_with_plasma(omega_I_gap, omega_cutoff, omega_j)

    m0 = 2.0 * epsilon / (1.0 - omega_gap**2 / omega_j**2)
    gamma = m0 / 4.0 * np.sqrt(kS0 * kI0)  # rad/cell

    dkS = _dk_dw(omega_gap, omega_cutoff, omega_j)
    dkI = _dk_dw(omega_I_gap, omega_cutoff, omega_j)

    delta_omega = 4.0 * gamma / (dkS + dkI)  # rad/GHz (same units as input)
    return omega_gap, delta_omega


def manual_solution(
    omegas: np.ndarray,
    omega_pump: float,
    omega_cutoff: float,
    omega_j: float,
    epsilon: float,
    ncell: int,
    cell_size: float,
    v_ratio: float,
) -> np.ndarray:
    """Backward-wave BVP using Bloch dispersion and v_ratio.

    Gap centre (linear approx): ωgap = ωp/2·(v_ratio - 1).
    Returns |S31|².
    """
    omegas = np.asarray(omegas, dtype=float)
    omega_I = omegas + omega_pump
    valid = omega_I < omega_cutoff
    omega_I_c = np.where(valid, omega_I, omega_cutoff * 0.9999)

    v_signal = cell_size * omega_cutoff / 2
    v_pump = v_signal / v_ratio

    kS = dispersion_linear(omegas, omega_cutoff)  # rad/cell (Bloch)
    kI = dispersion_linear(omega_I_c, omega_cutoff)  # rad/cell (Bloch)
    kp = omega_pump * cell_size / v_pump  # rad/cell

    kappa = -kp + kS + kI  # = Δk, rad/cell

    m_eff = 2.0 * epsilon / (1.0 - omegas**2 / omega_j**2)  # = 2·Zs_harm/Zs0

    q = -m_eff / 4 * np.emath.sqrt((kS - kappa) * (kI + kappa))  # rad/cell

    L = ncell  # cells (not metres)

    Delta = q**2 - (kappa / 2.0) ** 2
    Delta_sqrt = np.emath.sqrt(Delta)  # real inside gap

    return np.where(
        valid,
        np.abs(
            1
            / (
                np.cosh(Delta_sqrt * L)
                + 0.5j * kappa / Delta_sqrt * np.sinh(Delta_sqrt * L)
            )
        )
        ** 2,
        np.nan,
    ), np.zeros(len(omegas))