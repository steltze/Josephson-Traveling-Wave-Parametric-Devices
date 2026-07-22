from dataclasses import dataclass, field
import numpy as np
from logger import get_logger

log = get_logger(__name__)


@dataclass
class SimulationConfig:
    """
    All parameters needed for a TWPA/TWPC simulation.

    Parameters
    ----------
    M : int
        Floquet truncation order (number of pump harmonics).
    ks_state : list[int]
        Sideband indices tracked in the state vector.
        [0] → single-mode (signal only), [0, 1] → two-mode (signal + idler).
    Z0 : float
        Port impedance (Ω).
    ncell : int
        Number of unit cells.
    cell_size : float
        Cell length in metres (sets phase velocities).
    omega_cutoff : float
        Signal/idler mode cutoff angular frequency (rad/GHz).
    omega_j : float
        Junction plasma angular frequency (rad/GHz).
    epsilon : float
        Pump modulation depth of the series inductance.
    omega_c : float
        Target centre angular frequency for phase matching (rad/GHz).
    v_ratio : float
        v_signal / v_pump — how much slower the pump is vs the signal.
    omega_pump : float | None
        Pump angular frequency (rad/GHz). If None, derived as v_ratio * omega_c.
    freq_min, freq_max : float
        Frequency sweep edges (GHz).
    n_freqs : int
        Number of frequency points.
    disorder : bool
        Add random cell-to-cell parameter disorder.
    disorder_span : float
        Fractional spread of the uniform disorder distribution.
    disorder_seed : int | None
        RNG seed for reproducibility.
    nramp : int
        Number of cells over which the pump amplitude ramps up/down.
        0 disables the adiabatic envelope.
    """

    # Transfer matrix
    M: int = 1
    ks_state: list[int] = field(default_factory=lambda: [0, 1])

    # Port
    Z0: float = 50.0

    # Geometry
    ncell: int = 500
    cell_size: float = 10e-6

    # Circuit frequencies (rad/GHz)
    omega_cutoff: float = 50 * 2 * np.pi
    omega_j: float = 30 * 2 * np.pi

    # Pump
    epsilon: float = 0.0
    omega_c: float = 5 * 2 * np.pi
    v_ratio: float = 3.0
    omega_pump: float | None = None  # None → v_ratio * omega_c

    # Frequency sweep (GHz)
    freq_min: float = 0.5
    freq_max: float = 12.0
    n_freqs: int = 500

    # Disorder
    disorder: bool = False
    disorder_span: float = 0.01
    disorder_seed: int | None = 42

    # Adiabatic ramp (0 = flat)
    nramp: int = 0

    def __post_init__(self) -> None:
        if self.omega_pump is None:
            self.omega_pump = self.v_ratio * self.omega_c

        if self.ks_state:
            max_k = max(self.ks_state)
            max_sideband_GHz = self.freq_max + max_k * self.omega_pump / (2 * np.pi)
            max_sideband_omega = max_sideband_GHz * 2 * np.pi
            ratio = max_sideband_omega / self.omega_j
            if ratio > 0.5:
                log.warning(
                    "Max sideband frequency %.1f GHz is %.0f%% of ω_j = %.1f GHz. "
                    "The plasma resonance coupling factor 1/(1-ω²/ω_j²) will be "
                    "%.1fx — consider increasing ω_j or reducing freq_max or ω_pump.",
                    max_sideband_GHz,
                    ratio * 100,
                    self.omega_j / (2 * np.pi),
                    1 / (1 - ratio**2),
                )

        if self.M > 1 and self.ks_state:
            span = max(self.ks_state) - min(self.ks_state)
            if self.M > span:
                ks_min = min(self.ks_state)
                suggested = list(range(ks_min, ks_min + self.M + 1))
                log.warning(
                    "M=%d but ks_state=%s has span %d — harmonics of order %d..%d "
                    "have no in-state pairs to couple and will have no effect. "
                    "The result will be identical to M=%d. "
                    "To use all M=%d harmonics set ks_state=%s.",
                    self.M,
                    self.ks_state,
                    span,
                    span + 1,
                    self.M,
                    max(span, 1),
                    self.M,
                    suggested,
                )

    @property
    def freqs(self) -> np.ndarray:
        """Frequency array in GHz."""
        return np.linspace(self.freq_min, self.freq_max, self.n_freqs)

    @property
    def omegas(self) -> np.ndarray:
        """Angular frequency array in rad/GHz."""
        return self.freqs * 2 * np.pi

    @property
    def v_signal(self) -> float:
        """Signal phase velocity (cell_size * omega_cutoff / 2)."""
        return self.cell_size * self.omega_cutoff / 2

    @property
    def v_pump(self) -> float:
        """Pump phase velocity (m/s)."""
        return self.v_signal / self.v_ratio
    
    @property
    def propagation_direction(self) -> float:
        """Co- or counter-propagating signal and idler."""
        if self.v_ratio > 0:
            return 1.0
        else:
            return -1.0
