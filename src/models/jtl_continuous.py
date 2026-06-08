"""
Continuous-limit backward-wave coupled-mode model for the JTL TWPA/TWPC.

Physical picture
----------------
Pump travels LEFT (+kp direction in +x), signal travels RIGHT (+ks).
The pump opens a parametric stop-band (gap) at frequencies where:

    ks(ωs) + kI(ωI) = kp          (backward-wave phase matching)
    ωI = ωs + ωp                   (sideband / frequency matching)

Inside the gap the signal is converted to a BACKWARD-TRAVELING idler that
exits at the LEFT port (S21 in the 4-port S-matrix), not the right (S41).
The coupled-mode ODEs are:

    dEs/dx  = -(m_eff·kI/4)·EI     (signal driven by backward idler)
    -dEI/dx = +(m_eff·ks/4)·Es     (idler driven by signal)

These give EXPONENTIAL (cosh/sinh) solutions inside the gap — a stop-band.

Effective modulation depth
--------------------------
The JTL modulates the series inductance as

    Zs(ω,t) = Zs0(ω) + Zs_harm(ω)·e^{+iΘp} + Zs_harm(ω)·e^{-iΘp}

where Zs0 = iωL/(1-ω²/ωj²) and Zs_harm = iωL·ε/(1-ω²/ωj²)².
BOTH harmonics contribute to the coupling, giving m_eff = 2·Zs_harm/Zs0:

    m_eff(ω) ≈ 2ε/(1 - ω²/ωj²)

(The exact expression uses the geometric mean of the signal/idler corrections,
but this single-frequency approximation at ωs is within ~10% for typical params.)

S-parameter convention
----------------------
The ABCD→S conversion in this code uses voltage phasors normalized to Z0=50 Ω
for all ports. In this convention |S_ij|² is a VOLTAGE RATIO, not a photon
flux ratio. The correct energy-conservation check (pump included) is:

    Σᵢ (ωᵢ/ωₛ) |S_{i1}|²  =  1      (energy conserved with pump providing ωp per photon)
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq


def dispersion(omegas: np.ndarray, omega_cutoff: float) -> np.ndarray:
    """Bloch wavenumber k(ω) in rad/cell for the LC-ladder dispersion.

    k·a = 2·arcsin(ω/ωc),  valid for ω ≤ ωc.
    """
    return 2.0 * np.arcsin(
        np.clip(np.asarray(omegas, dtype=float) / omega_cutoff, 0.0, 1.0)
    )


def phase_mismatch(
    omegas: np.ndarray,
    omega_pump: float,
    omega_cutoff: float,
    v_pump: float,
    cell_size: float,
) -> np.ndarray:
    """Backward-wave phase mismatch Δk = ks + kI − kp  in rad/cell.

    Zero at the gap-centre frequency, positive above it, negative below.
    """
    omegas = np.asarray(omegas, dtype=float)
    omega_I = omegas + omega_pump
    kp = omega_pump * cell_size / v_pump  # rad/cell
    valid = omega_I < omega_cutoff
    kI = np.where(valid, dispersion(omega_I, omega_cutoff), np.nan)
    ks = dispersion(omegas, omega_cutoff)
    return ks + kI - kp


def gap_center(
    omega_pump: float,
    omega_cutoff: float,
    v_pump: float,
    cell_size: float,
) -> float | None:
    """Solve ks(ωgap) + kI(ωgap + ωp) = kp for ωgap.

    Returns None if no solution exists.
    """
    kp = omega_pump * cell_size / v_pump

    def residual(ws: float) -> float:
        omega_I = ws + omega_pump
        if omega_I >= omega_cutoff or ws <= 0.0:
            return float("nan")
        return dispersion(ws, omega_cutoff) + dispersion(omega_I, omega_cutoff) - kp

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

    gap_center_freq = omega_pump / 2.0 * (v_ratio - 1.0)
    delta = gap_center_freq - omegas  # detuning from gap centre
    v_signal = cell_size * omega_cutoff / 2

    kS = dispersion(omegas, omega_cutoff)  # rad/cell (Bloch)
    kI = dispersion(omega_I_c, omega_cutoff)  # rad/cell (Bloch)
    v_pump = v_signal / v_ratio
    kp = omega_pump * cell_size / v_pump  # rad/cell

    kappa = -kp + kS + kI  # = Δk, rad/cell

    # kappa = 2.0 * delta / v_signal * cell_size

    m = 2.0 * epsilon / (1.0 - omegas**2 / omega_j**2)

    q = -m / 4 * np.emath.sqrt((kS - kappa) * (kI + kappa))  # rad/cell

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
    )
