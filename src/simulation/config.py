from dataclasses import dataclass, field
import numpy as np


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
        Signal/idler mode cutoff angular frequency (rad/s).
    omega_j : float
        Junction plasma angular frequency (rad/s).
    epsilon : float
        Pump modulation depth of the series inductance.
    omega_c : float
        Target centre angular frequency for phase matching (rad/s).
    v_ratio : float
        v_sigma / v_pump — how much slower the pump is vs the signal.
    freq_min, freq_max : float
        Frequency sweep edges (Hz).
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

    # Circuit frequencies (rad/s)
    omega_cutoff: float = 50e9 * 2 * np.pi
    omega_j: float = 30e9 * 2 * np.pi

    # Pump
    epsilon: float = 0.0
    omega_c: float = 5e9 * 2 * np.pi
    v_ratio: float = 3.0

    # Frequency sweep (Hz)
    freq_min: float = 0.5e9
    freq_max: float = 12e9
    n_freqs: int = 500

    # Disorder
    disorder: bool = False
    disorder_span: float = 0.01
    disorder_seed: int | None = 42

    # Adiabatic ramp (0 = flat)
    nramp: int = 0

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
        """Signal phase velocity (m/s)."""
        return self.cell_size * self.omega_cutoff

    @property
    def v_pump(self) -> float:
        """Pump phase velocity (m/s)."""
        return self.v_sigma / self.v_ratio

    @property
    def omega_pump(self) -> float:
        """
        Phase-matched pump frequency (rad/s).

        Derived from the continuous-wave phase-matching condition
        k_p = omega_p / v_p = omega_c / v_sigma * v_ratio, giving
        omega_p = v_ratio * omega_c.
        """
        return self.v_ratio * self.omega_c
