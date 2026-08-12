
# ---------------------------------------------------------------------------
# Dispersion relations — uniform signature (omegas, omega_cutoff, omega_j)
# ---------------------------------------------------------------------------
import numpy as np

def dispersion_linear(
    omegas: np.ndarray, omega_cutoff: float, omega_j: float = np.inf
) -> np.ndarray:
    """k·a = 2ω/ωc  (pure linear, no JJ)."""
    return 2.0 * np.asarray(omegas, dtype=float) / omega_cutoff


def dispersion_linear_with_plasma(
    omegas: np.ndarray, omega_cutoff: float, omega_j: float
) -> np.ndarray:
    """k·a = 2ω/ωc / √(1 - ω²/ωj²)  (linear + JJ correction)."""
    w = np.asarray(omegas, dtype=float)
    return 2.0 * w / (omega_cutoff * np.sqrt(1.0 - (w / omega_j) ** 2))


def dispersion_bloch(
    omegas: np.ndarray, omega_cutoff: float, omega_j: float = np.inf
) -> np.ndarray:
    """k·a = 2·arcsin(ω/ωc)  (exact discrete LC ladder, no JJ)."""
    w = np.asarray(omegas, dtype=float)
    arg = 1.0 - 2.0 * (w / omega_cutoff) ** 2
    return np.arccos(arg)


def dispersion_bloch_with_plasma(
    omegas: np.ndarray, omega_cutoff: float, omega_j: float
) -> np.ndarray:
    """k·a = arccos(1 - 2(ω/ωc)² / (1 - ω²/ωj²))  (full discrete JTL)."""
    w = np.asarray(omegas, dtype=float)
    denom = 1.0 - (w / omega_j) ** 2
    arg = 1.0 - 2.0 * (w / omega_cutoff) ** 2 / denom
    return 
