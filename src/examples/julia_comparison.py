import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete
from analysis.checks import check_transfer_matrix_determinant
from examples.utils import save_all

log = get_logger(__name__)


def julia_comparison():
    ks_state = [0, 1, 2, 3, 4]
    M = 1
    ncell = 320+1
    cfg = SimulationConfig(
        Z0=50,
        M=M,
        ks_state=ks_state,
        ncell=ncell,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
        omega_pump=3.4 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,  # usually smaller
        epsilon=0.15,  # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
        omega_c=3.4
        * 2
        * np.pi,  # gap: well in the S parameter from signal to tranmission
        v_ratio=2.5,
        freq_min=1,  # GHz
        freq_max=14,  # GHz
        n_freqs=500,
        disorder=False,
        nramp=0,
    )

    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi))

    sim = Simulation(JTLDiscrete, cfg)
    sim.plot_s_parameters([(6, 1), (1, 6)], k=0, normalize=True)

    # sim.plot_s_parameters([(i, 2) for i in range(1, 2 * len(ks_state) + 1)], k=1, normalize=True)
    # sim.plot_s_parameters([(2, i) for i in range(1, 2 * len(ks_state) + 1)], k=1, normalize=True)

    S = sim.get_s_matrix().array

    log.info(f"f cutoff = {cfg.omega_cutoff / 2 / np.pi}")

    # S_matrix = sim.get_s_matrix().array
    T_grid = sim._get_T_grid()

    _, _ = check_transfer_matrix_determinant(T_grid, tolerance=1e-18)

    # symbolic_matrix, _ = sim.get_symbolic_matrix()
    # a = CellSingleMode()
    # a.export_matrix_graphic(symbolic_matrix)

    # save_all("julia_comparison")
    plt.show()

