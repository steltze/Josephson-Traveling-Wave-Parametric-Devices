"""
Unit cell definitions for Josephson transmission lines.

Each cell is defined by its series impedance Z_s(ω) and shunt admittance Y_g(ω):

    Z_s(ω) = Z_s0(ω) + ε · Z_s1(ω) · e^{iθ}
    Y_g(ω) = Y_g0(ω) + ε · Y_g1(ω) · e^{iθ}

The plasma frequency correction 1/(1 - ω²/ωJ²) accounts for the JJ capacitance
in parallel with the inductance, which bends the dispersion relation and sets the
high-frequency cutoff.

All arrays have shape (Nf, Nc) where Nf = number of frequencies, Nc = number of cells.

Notation:
s: series e.g. Zs
g: ground e.g. Yg
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Cell:
    """
    Physical parameters for a single unit cell.

    Parameters
    ----------
    Zs : int
        Series impedance per cell.

    Yg : int
        Shunt admittance to ground per cell.

    Yi : int
        Inner coupling admittance per cell.
        Only relevant for two-mode lines.
        If 0, the line degenerates to a single-mode line (Σ = Δ ports).

    epsilon_s : int
        Local pump modulation amplitude per cell.

    epsilon_g : int
        Local pump modulation amplitude per cell.

    theta : array-like
        Local pump phase per cell (rad).
        Computed from the pump wavevector: θ_n = k_p · n · a.
    """

    Zs: int
    Yg: int
    Yi: int
    epsilon_s: int
    epsilon_g: int
    theta: np.ndarray

    # @property
    # def omega_plasma(self) -> np.ndarray:
    #     """
    #     Junction plasma angular frequency ωJ = 1/sqrt(Ls·Cs) per cell (rad/s).
    #     Sets the high-frequency cutoff of the line.
    #     """
    #     return 1.0 / np.sqrt(self.Ls * self.Cs)

    @property
    def omega_sigma(self) -> np.ndarray:
        """
        Σ mode cutoff angular frequency per cell (rad/s).
        ω_Σ = 1/sqrt(Ls·Cg)
        """
        return 1.0 / np.sqrt(self.Ls * self.Cg)

    @property
    def omega_delta(self) -> np.ndarray:
        """
        Δ mode cutoff angular frequency per cell (rad/s).
        ω_Δ = 1/sqrt(Ls·(Cg + 2·Ci))
        """
        return 1.0 / np.sqrt(self.Ls * (self.Cg + 2 * self.Ci))

    @property
    def Z_sigma(self) -> np.ndarray:
        """Σ mode characteristic impedance sqrt(Ls/Cg) per cell (Ω)."""
        return np.sqrt(self.Ls / self.Cg)

    @property
    def Z_delta(self) -> np.ndarray:
        """Δ mode characteristic impedance sqrt(Ls/(Cg+2Ci)) per cell (Ω)."""
        return np.sqrt(self.Ls / (self.Cg + 2 * self.Ci))


def compute_impedances(
    cell: CellParameters,
    omega_signal: np.ndarray,
    omega_pump: float,
) -> dict:
    """
    Compute the series impedances and shunt admittances for signal and idler
    frequencies, including the plasma frequency correction and pump modulation.

    The plasma frequency correction comes from the JJ equivalent circuit:
    a linear inductance Ls in parallel with a capacitance Cs. The resulting
    impedance is:
        Z_s(ω) = jωL / (1 - ω²/ωJ²)

    The first-order pump-modulated correction (order ε) is:
        Z_s1(ω) = Z_s0(ω) · ε / (1 - ω²/ωJ²)

    This extra factor of 1/(1 - ω²/ωJ²) comes from expanding the inductance
    modulation to first order in ε while keeping the full plasma frequency
    denominator — see handwritten notes page 1.

    Parameters
    ----------
    cell : CellParameters
        Cell parameters for all Nc cells.

    omega_signal : np.ndarray, shape (Nf,)
        Signal angular frequencies (rad/s).

    omega_pump : float
        Pump angular frequency (rad/s).

    Returns
    -------
    dict with keys:
        Zs0_sigma : (Nf, Nc) — unperturbed series impedance at signal frequency
        Zs1_sigma : (Nf, Nc) — pump-modulated series impedance at signal frequency
        Zs0_delta : (Nf, Nc) — unperturbed series impedance at idler frequency
        Zs1_delta : (Nf, Nc) — pump-modulated series impedance at idler frequency
        Yg0_sigma : (Nf, Nc) — unperturbed shunt admittance at signal frequency
        Yg0_delta : (Nf, Nc) — unperturbed shunt admittance at idler frequency
        omega_idler : float — idler angular frequency
    """
    omega_idler = omega_signal + omega_pump  # energy conservation: ωI = ωS + ωP

    # Expand dims for broadcasting: (Nf, 1) * (1, Nc) → (Nf, Nc)
    ws = omega_signal[:, np.newaxis]  # (Nf, 1)
    wd = omega_idler[:, np.newaxis]  # (Nf, 1)

    Ls = cell.Ls[np.newaxis, :]  # (1, Nc)
    Cg = cell.Cg[np.newaxis, :]  # (1, Nc)
    Ci = cell.Ci[np.newaxis, :]  # (1, Nc)
    eps = cell.epsilon[np.newaxis, :]  # (1, Nc)

    # Plasma frequency per cell
    omega_J = cell.omega_plasma[np.newaxis, :]  # (1, Nc)

    # Plasma frequency correction denominators
    denom_sigma = 1 - ws**2 / omega_J**2  # (Nf, Nc)
    denom_delta = 1 - wd**2 / omega_J**2  # (Nf, Nc)

    # Series impedances (zeroth order)
    Zs0_sigma = 1j * ws * Ls / denom_sigma  # (Nf, Nc)
    Zs0_delta = 1j * wd * Ls / denom_delta  # (Nf, Nc)

    # Series impedances (first order in ε — pump modulated)
    # Extra 1/denom factor from expanding L(t) to first order
    Zs1_sigma = Zs0_sigma * eps / denom_sigma  # (Nf, Nc)
    Zs1_delta = Zs0_delta * eps / denom_delta  # (Nf, Nc)

    # Shunt admittances (no pump modulation — capacitors are linear)
    Yg0_sigma = 1j * ws * Cg  # (Nf, Nc)
    Yg0_delta = 1j * wd * (Cg + 2 * Ci)  # (Nf, Nc)

    return {
        "Zs0_sigma": Zs0_sigma,
        "Zs1_sigma": Zs1_sigma,
        "Zs0_delta": Zs0_delta,
        "Zs1_delta": Zs1_delta,
        "Yg0_sigma": Yg0_sigma,
        "Yg0_delta": Yg0_delta,
        "omega_idler": omega_idler,
    }


def build_cell_parameters_from_config(config) -> CellParameters:
    """
    Derive CellParameters from a SimulationConfig.

    This is the main entry point for building cell parameters. It:
    1. Derives L, C values from the mode frequencies and impedance
    2. Computes the pump phase profile (chirped or uniform)
    3. Applies the adiabatic ramp envelope to epsilon
    4. Optionally adds disorder

    Parameters
    ----------
    config : SimulationConfig

    Returns
    -------
    CellParameters
    """
    ncell = config.ncell
    ns = np.arange(ncell)

    # --- Derive circuit element values from mode frequencies ---
    # From: omega_sigma = 1/sqrt(L*Cg) and Z_sigma = sqrt(L/Cg) = Z0
    # Solving: L = Z0/omega_sigma, Cg = 1/(omega_sigma*Z0)
    ZRs = config.Z0 * np.ones(ncell)
    Lss = ZRs / config.omega_sigma0
    Cgs = 1.0 / (config.omega_sigma0 * ZRs)

    # From: omega_delta = 1/sqrt(L*(Cg+2*Ci))
    # Solving for Ci: Ci = Cg*(omega_sigma²/omega_delta² - 1) / 2
    rspeed = (config.omega_delta0 / config.omega_sigma0) ** 2
    Cis = (1 - rspeed) / (2 * rspeed) * Cgs

    # Junction capacitance from plasma frequency: Cs = 1/(omega_J² * L)
    Css = 1.0 / (config.omega_j0**2 * Lss)

    # --- Pump phase profile ---
    # Chirped pump velocity: varies linearly from chirp_xmin to chirp_xmax * v_pump0
    vps = np.linspace(
        config.chirp_xmin * config.v_pump0,
        config.chirp_xmax * config.v_pump0,
        ncell,
    )
    # theta_n = k_p * n * a = (omega_p / v_p(n)) * n * a
    # Positive sign: pump applied backward (counter-propagating)
    thetas = +config.omega_pump / vps * ns * config.cell_size

    # --- Epsilon profile with optional adiabatic ramp ---
    if config.nramp > 0:
        alpha = 4.0 / config.nramp
        ramp_up = 0.5 * (1 + np.tanh(alpha * (ns - config.nramp / 2)))
        ramp_down = 0.5 * (1 + np.tanh(alpha * ((ncell - 1 - config.nramp / 2) - ns)))
        profile = ramp_up * ramp_down
    else:
        profile = np.ones(ncell)

    epsilons = profile * config.epsilon

    # --- Optional disorder ---
    if config.disorder:
        rng = np.random.default_rng(config.disorder_seed)
        span = config.disorder_span
        Lss = Lss * rng.uniform(1 - span / 2, 1 + span / 2, ncell)
        Css = Css * rng.uniform(1 - span / 2, 1 + span / 2, ncell)
        Cgs = Cgs * rng.uniform(1 - span / 2, 1 + span / 2, ncell)

    return CellParameters(
        Ls=Lss,
        Cs=Css,
        Cg=Cgs,
        Ci=Cis,
        epsilon=epsilons,
        theta=thetas,
    )
