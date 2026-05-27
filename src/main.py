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
        ks_state=[0],
        ncell=500,
        cell_size=10e-6,
        omega_cutoff=50e9 * 2 * np.pi,
        omega_j=30e9 * 2 * np.pi,
        epsilon=0.0,
        omega_c=5e9 * 2 * np.pi,
        v_ratio=3.0,
        freq_min=0.5e9,
        freq_max=12e9,
        n_freqs=500,
        disorder=False,
        nramp=0,
    )

    logging.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi * 1e9))

    sim = Simulation(JTL, cfg)

    symbolic_matrix, _ = sim.get_symbolic_matrix()
    # a = CellSingleMode()
    # a.export_matrix_graphic(symbolic_matrix)
    # sim.plot_s_parameters([(3, 1), (1, 1), (2, 1), (4, 1)])
    sim.plot_s_parameters([(2, 1), (1, 1)])
    sim.plot_dispersion_relation()
    plt.show()


if __name__ == "__main__":
    main()
