import logging

import numpy as np
import matplotlib.pyplot as plt

from simulation import SimulationConfig, Simulation
from models import JTL
from symbolic import CellSingleMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0, 1],
        ncell=500,
        cell_size=10e-6,
        omega_cutoff=50e9 * 2 * np.pi,  # RH ATL
        # omega_cutoff=0.1e9 * 2 * np.pi, # LH ATL
        omega_pump=15e9 * 2 * np.pi,
        omega_j=60e9 * 2 * np.pi,
        epsilon=0.2,
        omega_c=5e9 * 2 * np.pi,
        v_ratio=2.0,
        freq_min=1e9,
        freq_max=12e9,
        n_freqs=500,
        disorder=False,
        nramp=0.0,
    )

    logging.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi * 1e9))

    # sim = Simulation(JTL.left_handed(), cfg)
    sim = Simulation(JTL, cfg)

    symbolic_matrix, _ = sim.get_symbolic_matrix()
    # a = CellSingleMode()
    # a.export_matrix_graphic(symbolic_matrix)
    # sim.plot_s_parameters([(3, 1), (1, 1), (2, 1), (4, 1)])
    sim.plot_s_parameters([(1, 1), (2, 1), (3, 1), (4, 1)])
    sim.plot_s_parameters([(1, 1), (1, 2), (1, 3), (1, 4)])
    sim.plot_dispersion_relation()
    plt.show()


def main2():
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0, 1],
        ncell=320,
        cell_size=10e-6,
        omega_cutoff=50 / 530e-12,        # L = 530 pH, C = 212 fF → ~15 GHz
        omega_pump=6.8e9 * 2 * np.pi,
        omega_j=60e9 * 2 * np.pi,
        epsilon=0.1,                    # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
        omega_c=3.4e9 * 2 * np.pi,
        v_ratio=2.5,
        freq_min=1e9,
        freq_max=12e9,
        n_freqs=500,
        disorder=False,
        nramp=0,
    )

    logging.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi * 1e9))

    sim = Simulation(JTL, cfg)

    sim.plot_s_parameters([(1, 1), (2, 1), (3, 1), (4, 1)])
    sim.plot_s_parameters([(1, 1), (1, 2), (1, 3), (1, 4)])
    sim.plot_dispersion_relation()

    omega_signals = cfg.omegas                          # rad/s
    omega_idlers = omega_signals + cfg.omega_pump       # rad/s
    S_matrix = np.abs(sim.get_s_matrix().array)
    check1 = (S_matrix[:, 0, 0]**2
              + (omega_signals / omega_idlers) * S_matrix[:, 1, 0]**2
              + S_matrix[:, 2, 0]**2
              + (omega_signals / omega_idlers) * S_matrix[:, 3, 0]**2)
    plt.figure()
    plt.plot(omega_signals, check1)
    print(check1)
    plt.show()


if __name__ == "__main__":
    main2()
