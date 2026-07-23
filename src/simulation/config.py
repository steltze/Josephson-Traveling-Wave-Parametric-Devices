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
    epsilon : float | list[float]
        Pump modulation depth of the series inductance. A single pump tone
        (float) or, for a multi-pump line, one depth per pump (length P) —
        see models.jtl_discrete_multipump.JTLDiscreteMultiPump.
    omega_c : float
        Target centre angular frequency for phase matching (rad/GHz).
    v_ratio : float | list[float]
        v_signal / v_pump — how much slower the pump is vs the signal.
        Float for single-pump, length-P list matching `epsilon` for
        multi-pump.
    omega_pump : float | list[float] | None
        Pump angular frequency (rad/GHz), float or length-P list matching
        `epsilon`. If None, derived elementwise as v_ratio * omega_c.
    Kmax : list[tuple[int, int]] | None
        Multi-pump only: per-pump (k_min, k_max) sideband range (length P,
        matching `epsilon`), e.g. [(-2, 3), (-1, 1)] -- need not be
        symmetric about 0. Replaces `ks_state`'s role for the multi-pump
        tensor lattice: n_sidebands[j] = k_max_j - k_min_j + 1. Unused for
        single-pump configs.
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

    # Pump (float = single tone; list[float] length P = multi-pump)
    epsilon: float | list[float] = 0.0
    omega_c: float = 5 * 2 * np.pi
    v_ratio: float | list[float] = 3.0
    omega_pump: float | list[float] | None = None  # None → v_ratio * omega_c
    Kmax: list[tuple[int, int]] | None = None  # multi-pump per-pump (k_min, k_max)

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
            if isinstance(self.v_ratio, list):
                self.omega_pump = [vr * self.omega_c for vr in self.v_ratio]
            else:
                self.omega_pump = self.v_ratio * self.omega_c

        multi_pump = isinstance(self.epsilon, list)
        if multi_pump:
            P = len(self.epsilon)
            for name, value in (("v_ratio", self.v_ratio), ("omega_pump", self.omega_pump)):
                if not isinstance(value, list) or len(value) != P:
                    raise ValueError(
                        f"{name} must be a list of length {P} (matching epsilon) "
                        "for a multi-pump config"
                    )
            if self.Kmax is not None:
                if not isinstance(self.Kmax, list) or len(self.Kmax) != P:
                    raise ValueError(f"Kmax must be a list of length {P} (matching epsilon)")
                if any(k_min > k_max for k_min, k_max in self.Kmax):
                    raise ValueError(f"each Kmax entry must have k_min <= k_max, got {self.Kmax}")
            return  # ks_state / plasma-resonance warnings below are single-pump-only

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
    def v_pump(self) -> float | list[float]:
        """Pump phase velocity (m/s), or one per pump for a multi-pump config."""
        if isinstance(self.v_ratio, list):
            return [self.v_signal / vr for vr in self.v_ratio]
        return self.v_signal / self.v_ratio
    
    @property
    def propagation_direction(self) -> float:
        """Co- or counter-propagating signal and idler."""
        v_ratio = self.v_ratio[0] if isinstance(self.v_ratio, list) else self.v_ratio
        if v_ratio > 0:
            return 1.0
        else:
            return -1.0
