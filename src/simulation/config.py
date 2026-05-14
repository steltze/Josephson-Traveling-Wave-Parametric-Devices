"""
SimulationConfig: single entry point for all simulation parameters.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationConfig:
    """
    Configuration for a Josephson transmission line simulation.

    The simulator accepts physical circuit parameters and derives everything
    else.

    Parameters
    ----------
    Z0 : float
        Port impedance in Ohms. Default 50 Ω.

    ncell : int
        Number of unit cells. More cells → sharper, deeper isolation gap
        but slower simulation.

    omega_sigma0 : float
        Σ mode cutoff angular frequency (rad/s).
        Sets the signal/idler propagation speed: v_sigma = a * omega_sigma0.

    omega_delta0 : float
        Δ mode cutoff angular frequency (rad/s).
        Sets the pump propagation speed: v_delta = a * omega_delta0.
        If equal to omega_sigma0, Ci = 0 (single-mode limit).

    omega_j0 : float
        Junction plasma angular frequency (rad/s).
        Sets the high-frequency cutoff of the line.

    epsilon : float
        Pump modulation amplitude (dimensionless). Assumed small (weak pump).
        Convention: L(x,t) = L0 * (1 + 2*epsilon*cos(wp*t)).

    v_ratio : float
        Ratio v_sigma / v_pump. Controls how slow the pump is relative to the signal.
        Higher ratio → easier phase matching → broader bandwidth.
        Default 3.0 (as in the paper).

    freq_min, freq_max : float
        Frequency sweep range in Hz (not rad/s — for human readability).

    n_freqs : int
        Number of frequency points in the sweep.

    """

    # Port impedance
    Z0: float = 50.0

    # Line geometry
    ncell: int = 500
    cell_size: float = 10e-6  # meters

    # Mode frequencies (rad/s)
    omega_sigma0: float = 50e9 * 2 * np.pi
    omega_delta0: float = 50e9 * 2 * np.pi
    omega_j0: float = 30e9 * 2 * np.pi

    # Pump
    epsilon: float = 0.2
    omega_c: float = 5e9 * 2 * np.pi
    v_ratio: float = 3.0  # v_sigma / v_pump

    # Frequency sweep (Hz, not rad/s)
    freq_min: float = 0.5e9
    freq_max: float = 12e9
    n_freqs: int = 2000

    # Random seed for reproducibility (None = random each time)
    disorder_seed: int | None = None

    def __post_init__(self):
        """Validate config."""

    @property
    def freqs(self) -> np.ndarray:
        """Frequency array in Hz."""
        return np.linspace(self.freq_min, self.freq_max, self.n_freqs)

    @property
    def omegas(self) -> np.ndarray:
        """Angular frequency array in rad/s."""
        return self.freqs * 2 * np.pi

    @property
    def v_sigma(self) -> float:
        """Σ mode phase velocity (m/s)."""
        return self.cell_size * self.omega_sigma0

    @property
    def v_delta(self) -> float:
        """Δ mode phase velocity (m/s)."""
        return self.cell_size * self.omega_delta0

    @property
    def v_pump0(self) -> float:
        """Nominal pump phase velocity (m/s), derived from v_ratio."""
        return self.v_sigma / self.v_ratio

    @property
    def omega_pump(self) -> float:
        """
        Pump angular frequency derived from phase matching condition at omega_c.

        Assumes v_I = v_S.
        """
        v_s = self.v_sigma
        v_d = self.v_delta
        v_p = self.v_pump0
        omega_p = (v_d / v_s + 1) / (v_d / v_p - 1) * self.omega_c
        if omega_p < 0:
            raise ValueError(
                f"Derived omega_pump={omega_p:.3e} is negative. "
                "Check v_ratio and omega_c — phase matching may not be achievable "
                "with these parameters."
            )
        return omega_p