import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from analysis.checks import (
    check_transfer_matrix_determinant,
)
from logger import get_logger
from models import JTLDiscrete
from simulation import Simulation, SimulationConfig

log = get_logger(__name__)


def harmonics_comparison():
    ncell = 320
    freq_min = 1  # GHz
    freq_max = 12  # GHz
    n_freqs = 200
    signal_freqs = np.linspace(freq_min, freq_max, n_freqs) * 2 * np.pi

    plt.figure()

    db = np.zeros((3, len(signal_freqs)))
    energy_checks = np.zeros((2, len(signal_freqs)))

    for index, (M, ks_state, (i, j)) in enumerate(
        [
            (1, [-2, -1, 0, 1, 2], (7, 2)),
            (2, [-2, -1, 0, 1, 2], (7, 2)),
            (5, [-2, -1, 0, 1, 2], (7, 2)),
        ]
    ):
        cfg = SimulationConfig(
            Z0=50,
            M=M,
            ks_state=ks_state,
            ncell=ncell,
            cell_size=10e-6,
            omega_cutoff=2 * 50 / 530e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
            omega_pump=6.8 * 2 * np.pi,
            omega_j=60 * 2 * np.pi,  # usually smaller
            epsilon=0.06,  # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
            v_ratio=2.5,
            freq_min=freq_min,  # GHz
            freq_max=freq_max,  # GHz
            n_freqs=n_freqs,
            disorder=False,
            epsilon_nramp=0,
        )

        sim = Simulation(JTLDiscrete, cfg)
        T_grid = sim.get_transfer_matrix_grid()
        _, _ = check_transfer_matrix_determinant(T_grid, tolerance=1e-12)

        S_matrix = sim.get_s_matrix().array
        # _ = check_photon_conservation(S_matrix, cfg.omegas, cfg.omega_pump, cfg.ks_state)
        # energy_checks[index] = check_photon_flux_conservation(
        #     S_matrix, cfg.omegas, cfg.omega_pump, cfg.ks_state
        # )[:, j]

        db[index] = 20.0 * np.log10(np.abs(S_matrix[:, i, j]))

    plt.figure()
    # plt.plot(signal_freqs / 2 / np.pi, db[0], label="M=1")
    plt.plot(signal_freqs / 2 / np.pi, db[1], label="M=2")
    plt.plot(signal_freqs / 2 / np.pi, db[2], label="M=3")
    plt.legend()

    # plt.figure()
    # plt.plot(signal_freqs / 2 / np.pi, energy_checks[0], label="M=1")
    # plt.plot(signal_freqs / 2 / np.pi, energy_checks[1], label="M=2")
    # plt.plot(signal_freqs / 2 / np.pi, energy_checks[2], label="M=3")
    # plt.legend()

    print(np.abs(db[1] - db[0]).sum())
    print(np.abs(energy_checks[1] - energy_checks[0]).sum())

    # save_all("harmonics_comparison")
    plt.show()
