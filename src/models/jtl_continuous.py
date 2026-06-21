"""
Continuous coupled-mode backward-wave model for the JTL TWPA/TWPC.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from solver.s_matrix import SMatrix
from analysis.s_parameters import plot_s_parameters as _plot_s_params
from logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dispersion relations — uniform signature (omegas, omega_cutoff, omega_j)
# ---------------------------------------------------------------------------


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
    return 2.0 * np.arcsin(
        np.clip(np.asarray(omegas, dtype=float) / omega_cutoff, 0.0, 1.0)
    )


def dispersion_bloch_with_plasma(
    omegas: np.ndarray, omega_cutoff: float, omega_j: float
) -> np.ndarray:
    """k·a = arccos(1 - 2(ω/ωc)² / (1 - ω²/ωj²))  (full discrete JTL)."""
    w = np.asarray(omegas, dtype=float)
    denom = 1.0 - (w / omega_j) ** 2
    arg = np.clip(1.0 - 2.0 * (w / omega_cutoff) ** 2 / denom, -1.0, 1.0)
    return np.arccos(arg)


# ---------------------------------------------------------------------------
# JTLContinuous class
# ---------------------------------------------------------------------------


class JTLContinuous:
    """
    Continuous coupled-mode backward-wave TWPA/TWPC model.

    Provides the same public interface as Simulation (get_s_matrix,
    plot_s_parameters) so it can be used alongside JTLDiscrete results.

    The dispersion relation used for signal/idler k-vectors is swappable.
    Pass it as a constructor argument or use with_dispersion() for a subclass:

        sim = JTLContinuous(cfg)
        sim = JTLContinuous(cfg, dispersion_fn=dispersion_bloch)
        sim = JTLContinuous.with_dispersion(dispersion_linear)(cfg)

    Dispersion options (all in models/jtl_continuous.py):
        dispersion_linear               k = 2ω/ωc
        dispersion_linear_with_plasma   k = 2ω/ωc / √(1-ω²/ωj²)
        dispersion_bloch                k = 2·arcsin(ω/ωc)
        dispersion_bloch_with_plasma    k = arccos(1-2(ω/ωc)²/(1-ω²/ωj²))
    """

    dispersion_fn = staticmethod(dispersion_bloch_with_plasma)

    _DISPERSION_LABELS = {
        "dispersion_linear": "linear dispersion",
        "dispersion_linear_with_plasma": "linear dispersion with plasma correction",
        "dispersion_bloch": "Bloch dispersion",
        "dispersion_bloch_with_plasma": "Bloch dispersion with plasma correction",
    }

    def __init__(self, config, dispersion_fn=None) -> None:
        self._cfg = config
        self._S_cache: SMatrix | None = None
        if dispersion_fn is not None:
            self.dispersion_fn = dispersion_fn
        fn = self.dispersion_fn
        label = self._DISPERSION_LABELS.get(fn.__name__, fn.__name__)
        _log.info("Using %s on the continuous model.", label)

    @classmethod
    def with_dispersion(cls, fn) -> type:
        """Return a subclass that uses *fn* as its dispersion relation."""

        class _Variant(cls):
            dispersion_fn = staticmethod(fn)

        _Variant.__name__ = f"JTLContinuous[{fn.__name__}]"
        return _Variant

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _k(self, omegas: np.ndarray) -> np.ndarray:
        return self.dispersion_fn(omegas, self._cfg.omega_cutoff, self._cfg.omega_j)

    def _dk(self, omega: float) -> float:
        """Numerical d(k·a)/dω via central difference."""
        eps = max(abs(omega) * 1e-7, 1e-10)
        return float(
            (self._k(np.float64(omega + eps)) - self._k(np.float64(omega - eps)))
            / (2.0 * eps)
        )

    # ------------------------------------------------------------------
    # Analytical quantities
    # ------------------------------------------------------------------

    def phase_mismatch(self, omegas: np.ndarray | None = None) -> np.ndarray:
        """Δk = ks + kI - kp in rad/cell. Zero at the gap centre."""
        cfg = self._cfg
        if omegas is None:
            omegas = cfg.omegas
        omegas = np.asarray(omegas, dtype=float)
        kp = cfg.omega_pump * cfg.cell_size / cfg.v_pump
        return self._k(omegas) + self._k(omegas + cfg.omega_pump) - kp

    def gap_center(self) -> float | None:
        """Solve ks(ωg) + kI(ωg+ωp) = kp. Returns None if no solution."""
        cfg = self._cfg
        kp = cfg.omega_pump * cfg.cell_size / cfg.v_pump

        def residual(ws: float) -> float:
            if ws <= 0.0:
                return float("nan")
            return float(
                self._k(np.float64(ws)) + self._k(np.float64(ws + cfg.omega_pump)) - kp
            )

        lo = cfg.omega_cutoff * 1e-4
        hi = (cfg.omega_cutoff - cfg.omega_pump) * 0.999
        if hi <= lo:
            return None
        r_lo, r_hi = residual(lo), residual(hi)
        if not (np.isfinite(r_lo) and np.isfinite(r_hi)):
            return None
        if r_lo * r_hi > 0:
            return None
        return brentq(residual, lo, hi)

    def gap_bandwidth(self) -> tuple[float | None, float | None]:
        """Analytical stop-band bandwidth Δω = 4γ / (dks/dω + dkI/dω)."""
        cfg = self._cfg
        omega_gap = self.gap_center()
        if omega_gap is None:
            return None, None

        omega_I_gap = omega_gap + cfg.omega_pump
        kS0 = float(self._k(np.float64(omega_gap)))
        kI0 = float(self._k(np.float64(omega_I_gap)))

        m0 = 2.0 * cfg.epsilon / (1.0 - omega_gap**2 / cfg.omega_j**2)
        gamma = m0 / 4.0 * np.sqrt(kS0 * kI0)

        delta_omega = 4.0 * gamma / (self._dk(omega_gap) + self._dk(omega_I_gap))
        return omega_gap, delta_omega

    # ------------------------------------------------------------------
    # S-matrix (BVP solution)
    # ------------------------------------------------------------------

    def get_s_matrix(self) -> SMatrix:
        """Compute and cache the S-matrix from the coupled-mode BVP."""
        if self._S_cache is None:
            self._S_cache = self._compute_s_matrix()
        return self._S_cache

    def _compute_s_matrix(self) -> SMatrix:
        cfg = self._cfg
        omegas = cfg.omegas
        omega_I = omegas + cfg.omega_pump
        valid = omega_I < cfg.omega_cutoff
        omega_I_c = np.where(valid, omega_I, cfg.omega_cutoff * 0.9999)

        kS = self._k(omegas)
        kI = self._k(omega_I_c)
        kp = cfg.omega_pump * cfg.cell_size / cfg.v_pump
        kappa = kS + kI - kp  # ks + kp = ki but kp, ki < 0 so we do -kp, -(-ki)

        m_eff = 2.1 * cfg.epsilon / (1.0 - omegas**2 / cfg.omega_j**2)
        # m_eff = 2.0 * cfg.epsilon

        q = (
            -m_eff / 4.0 * np.emath.sqrt((kS) * (kI))
        )  # no corrections added, (kS - kappa) * (kI + kappa)

        N = cfg.ncell
        Delta = q**2 - (kappa / 2.0) ** 2
        Delta_sqrt = np.emath.sqrt(Delta)

        S31 = np.where(
            valid,
            np.exp(-0.5j * kappa * N)
            / (
                np.cosh(Delta_sqrt * N)
                + 0.5j * kappa / Delta_sqrt * np.sinh(Delta_sqrt * N)
            ),
            np.nan + 0j,
        )

        gS = m_eff / 4.0 * kS  # no correction added, kI / (kI + kappa)
        S21 = np.where(
            valid,
            np.exp(0.5j * kappa * N) * gS / Delta_sqrt * np.sinh(Delta_sqrt * N) * S31,
            np.nan + 0j,
        )

        Nf = len(omegas)
        data = np.zeros((Nf, 4, 4), dtype=complex)
        data[:, 2, 0] = S31
        data[:, 1, 0] = S21 * np.sqrt(kI / kS)
        return SMatrix(data, cfg.Z0)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_s_parameters(self, params, ax=None, **kwargs):
        """Plot selected S-parameters in dB. params: list of (i,j) 1-based pairs."""
        return _plot_s_params(
            self.get_s_matrix().array, self._cfg.freqs, params, ax=ax, **kwargs
        )
