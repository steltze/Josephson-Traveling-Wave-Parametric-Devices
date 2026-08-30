from dataclasses import dataclass, field
import numpy as np
from scipy.special import jv
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
        used by models.jtl_discrete_multipump.JTLDiscreteMultiPump and
        models.jtl_discrete_slot_mode.JTLDiscreteSlotMode. JTLDiscrete
        does NOT use this field -- it takes its pump amplitude from
        `phi_rf_frac`/`phi_dc_frac` instead (see below), the physical
        flux-fraction parametrization, rather than mixing the two
        conventions.
    phi_rf_frac : float
        Physical pump AC flux drive, Phi_RF / Phi0 (mirrors
        JosephsonCircuits.jl's `Phi_ac_frac`). JTLDiscrete's pump
        amplitude -- together with `phi_dc_frac`, sets the exact
        Jacobi-Anger/Bessel-function coefficients of the pump-driven
        junction's critical current (see ModulatedInductor): `eps` in that
        expansion is `pi*phi_rf_frac`.
    phi_dc_frac : float
        SQUID DC flux bias point, Phi_dc / Phi0 (mirrors JosephsonCircuits.jl's
        `Phi_dc_frac`). Together with `phi_rf_frac` this sets the exact
        Jacobi-Anger/Bessel-function coefficients of the pump-driven
        junction's critical current (see ModulatedInductor): self-Kerr
        (pump renormalizing its own mean inductance) scales as
        cos(pi*phi_dc_frac), and the 3-wave-mixing strength as
        sin(pi*phi_dc_frac) -- zero at zero bias, maximal at 0.25
        (quarter flux quantum, the standard 3WM operating point).
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
        Max disorder as a percentage of L/C (e.g. 5 -> uniform +/-5%).
    disorder_seed : int | None
        RNG seed for reproducibility.
    epsilon_nramp : int
        Number of cells over which the pump amplitude ramps up/down.
        0 disables the adiabatic envelope.
    adiabatic_pump : bool
        If True, `v_pump` is an `ncell`-length ramp instead of a scalar,
        from `v_pump_ramp_start` to `v_pump_ramp_end` times the nominal
        pump velocity.
    v_pump_ramp_start, v_pump_ramp_end : float
        Endpoints of the `v_pump` ramp (as a fraction of nominal pump
        velocity), applied when `adiabatic_pump` is True.
    v_pump_ramp_power : float
        Shape exponent for the ramp's cell-index axis. 1 = linear; <1
        makes v_pump fall faster early (levelling off later); >1 makes
        it fall faster late (staying near ramp_start longer first).
    include_slot_modes : bool

    omega_cutoff_slot_mode : float
        Slot-line mode cutoff angular frequency (rad/GHz) -- sets the slot
        line's own per-cell Ls, Co the same way `omega_cutoff` sets the
        main JTL's L, C. See models.jtl_discrete_slot_mode.JTLDiscreteSlotMode.
    omega_cutoff_coupling : float
        Cutoff angular frequency (rad/GHz) for the main-line/slot-line
        coupling capacitor Ci, using the same L-C-ladder formula as
        `omega_cutoff` / `omega_cutoff_slot_mode`. Independent of both --
        it sets how strongly the two lines couple, not how either is
        detuned, so sweep it separately from `omega_cutoff_slot_mode`.
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
    phi_rf_frac: float = 0.05
    phi_dc_frac: float = 1/3
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
    disorder_span: float = 1e-1  # e.g 1%
    disorder_seed: int | None = 42

    # Adiabatic ramp (0 = flat) for epsilon
    epsilon_nramp: int = 0

    # Adiabatic pump velocity
    adiabatic_pump: bool = False
    v_pump_ramp_start: float = 1.2  # v_pump(cell 0) / nominal v_pump
    v_pump_ramp_end: float = 0.9  # v_pump(last cell) / nominal v_pump

    # Slot mode parameters
    include_slot_modes: bool = False
    omega_cutoff_slot_mode: float = 50 * 2 * np.pi
    omega_cutoff_coupling: float = 50 * 2 * np.pi

    def __post_init__(self) -> None:
        if self.omega_pump is None:
            if isinstance(self.v_ratio, list):
                self.omega_pump = [vr * self.omega_c for vr in self.v_ratio]
            else:
                self.omega_pump = self.v_ratio * self.omega_c

        multi_pump = isinstance(self.epsilon, list)
        if multi_pump:
            P = len(self.epsilon)
            for name, value in (
                ("v_ratio", self.v_ratio),
                ("omega_pump", self.omega_pump),
            ):
                if not isinstance(value, list) or len(value) != P:
                    raise ValueError(
                        f"{name} must be a list of length {P} (matching epsilon) "
                        "for a multi-pump config"
                    )
            if self.Kmax is not None:
                if not isinstance(self.Kmax, list) or len(self.Kmax) != P:
                    raise ValueError(
                        f"Kmax must be a list of length {P} (matching epsilon)"
                    )
                if any(k_min > k_max for k_min, k_max in self.Kmax):
                    raise ValueError(
                        f"each Kmax entry must have k_min <= k_max, got {self.Kmax}"
                    )
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
        """Pump phase velocity (m/s), or one per pump for a multi-pump config.

        If `adiabatic_pump`, each scalar is replaced by an `ncell`-length
        ramp ranging from `v_pump_ramp_start` to `v_pump_ramp_end` times
        the nominal velocity.
        """

        def v_of(vr: float) -> float | np.ndarray:
            v_nominal = self.v_signal / vr
            ramp_start, ramp_end = self.v_pump_ramp_start, self.v_pump_ramp_end
            if not self.adiabatic_pump:
                return v_nominal

            db = -1
            gain = np.arccosh(10 ** (-db / 20))
            vs = self.v_signal
            b = np.pi * self.phi_rf_frac
            self.epsilon = 2 * np.tan(np.pi * self.phi_dc_frac) * jv(1, b) / jv(0, b)
            print(self.epsilon)
            def vp_of(omega_s: np.ndarray) -> np.ndarray:
                return -vs / (2 * omega_s / self.omega_pump + 1)

            def attenuation_of(omega_s: np.ndarray, vp: np.ndarray) -> np.ndarray:
                return (
                    -(self.epsilon**2)
                    / 4
                    * (omega_s**2 / vs**2 + self.omega_pump * omega_s / vp / vs)
                )

            step_frac = 0.01
            f_lo, f_hi = 3.5, 5.0  # GHz
            omega_hi = f_hi * 2 * np.pi
            omega_targets = [f_lo * 2 * np.pi]
            while omega_targets[-1] < omega_hi:
                w = omega_targets[-1]
                delta_omega = 2 * vs * np.sqrt(attenuation_of(w, vp_of(w)))
                if not np.isfinite(delta_omega) or delta_omega <= 0:
                    raise ValueError(
                        f"Non-positive/invalid gain bandwidth at ω/2π={w / (2 * np.pi):.3f} GHz "
                        "-- check epsilon/omega_pump for this target range."
                    )
                w_next = w + delta_omega * step_frac
                if w_next >= omega_hi:
                    break
                omega_targets.append(w_next)
            if omega_targets[-1] < omega_hi - 1e-9:
                omega_targets.append(omega_hi)
            omega_s_target = np.array(omega_targets)

            # print(omega_s_target/2/np.pi)

            vp = vp_of(omega_s_target)
            attenuation = attenuation_of(omega_s_target, vp)
            ncells = np.ceil(gain / np.sqrt(attenuation) / self.cell_size).astype(int)

            log.info(f"Ncell list = {ncells}")

            self.ncell = int(ncells.sum())
            max_allowed_cells = 2500
            if self.ncell > max_allowed_cells:
                log.critical(f"Ncell = {self.ncell} > {max_allowed_cells}, exiting...")
                exit()
            log.info(f"Ncell list = {ncells.sum()}")
            vp_profile = np.repeat(vp, ncells)
            # import matplotlib.pyplot as plt
            # plt.plot(np.arange(self.ncell), vs/vp_profile)
            # plt.show()
            # exit()
            return vp_profile
            # return np.linspace(ramp_start * v_nominal, ramp_end * v_nominal, self.ncell)

        if isinstance(self.v_ratio, list):
            return [v_of(vr) for vr in self.v_ratio]
        return v_of(self.v_ratio)

    @property
    def propagation_direction(self) -> float:
        """Co- or counter-propagating signal and idler."""
        v_ratio = self.v_ratio[0] if isinstance(self.v_ratio, list) else self.v_ratio
        if v_ratio > 0:
            return 1.0
        else:
            return -1.0
