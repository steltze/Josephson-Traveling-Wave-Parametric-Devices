import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete
from models.epsilon_calibration import calibrate_epsilon_ramp
from examples.utils import save_all

log = get_logger(__name__)


def adiabatic_pump_calibration():
    """
    A chirped v_pump ramp (adiabatic_pump=True) sweeps the local
    phase-matched gap through a range of v_ratio. Left uncompensated, the
    achievable gap depth at fixed epsilon depends on v_ratio steeply and
    non-trivially, so the composite gap ends up much deeper at one end of
    the ramp than the other. epsilon_calibration=True (the default)
    adaptively tunes epsilon(z) against the real simulator -- not an
    extrapolated fit -- to flatten it; see models/epsilon_calibration.py.

    Compares S31 with and without that calibration, and plots the
    resulting epsilon(z) profile.
    """
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0, 1],
        ncell=320,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.5,
        omega_c=3.4 * 2 * np.pi,
        v_ratio=-2.5,  # < 0 => counter-propagating
        freq_min=1,
        freq_max=12,
        n_freqs=500,
        disorder=False,
        epsilon_nramp=0,
        adiabatic_pump=True,
        epsilon_calibration=False,
    )
    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi))

    cfg_uncompensated = replace(cfg, epsilon_calibration=False)
    cfg_compensated = replace(
        cfg, epsilon_calibration=False, epsilon_calibration_target_db=-40.6
    )

    runs = {
        "uncompensated (uniform epsilon)": cfg_uncompensated,
        "calibrated epsilon(z)": cfg_compensated,
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, run_cfg in runs.items():
        sim = Simulation(JTLDiscrete, run_cfg, backend="numpy")
        S = sim.get_s_matrix(normalize=True).array
        S31_db = 10 * np.log10(np.abs(S[:, 2, 0]) ** 2 + 1e-15)
        ax.plot(run_cfg.freqs, S31_db, label=label)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S31|^2 (dB)")
    ax.set_title(f"Chirped v_pump gap: calibrated vs. uncompensated epsilon, ncells={cfg.ncell}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # What the calibration actually did to epsilon(z).
    # eps_profile = calibrate_epsilon_ramp(cfg, target_depth_db=-40.6)
    # fig2, ax2 = plt.subplots(figsize=(9, 4))
    # ax2.plot(np.arange(cfg.ncell), eps_profile)
    # ax2.axhline(cfg.epsilon, color="gray", ls="--", lw=1, label="uncompensated (uniform) epsilon")
    # ax2.set_xlabel("Cell index")
    # ax2.set_ylabel("Calibrated epsilon(z)")
    # ax2.set_title("Adaptive epsilon(z) profile along the v_pump chirp")
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)
    # fig2.tight_layout()

    save_all("adiabatic_pump_calibration")
    plt.show()


if __name__ == "__main__":
    setup_logging()
    adiabatic_pump_calibration()
