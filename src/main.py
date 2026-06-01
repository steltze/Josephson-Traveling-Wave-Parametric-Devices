import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging

log = get_logger(__name__)
from simulation import SimulationConfig, Simulation
from models import JTL
from symbolic import CellSingleMode
from analysis.checks import check_photon_conservation, check_transfer_matrix_determinant


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

    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi * 1e9))

    # sim = Simulation(JTL.left_handed(), cfg)
    sim = Simulation(JTL, cfg)

    # symbolic_matrix, _ = sim.get_symbolic_matrix()
    # a = CellSingleMode()
    # a.export_matrix_graphic(symbolic_matrix)
    sim.plot_s_parameters([(1, 1), (2, 1), (3, 1), (4, 1)])
    sim.plot_s_parameters([(1, 1), (1, 2), (1, 3), (1, 4)])
    sim.plot_dispersion_relation()
    plt.show()


def julia_comparison():
    ks_state = [0, 1]
    ncell = 320
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=ks_state,
        ncell=ncell,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 530e-3,  # L = 530 pH, C = 212 fF → ~30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,  # usually smaller
        epsilon=0.055,  # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
        omega_c=3.4 * 2 * np.pi,
        v_ratio=2.5,
        freq_min=1,
        freq_max=12,
        n_freqs=1000,
        disorder=False,
        nramp=0,
    )

    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi))

    sim = Simulation(JTL, cfg)
    sim.plot_s_parameters([(i, 1) for i in range(1, 2 * len(ks_state) + 1)])
    sim.plot_s_parameters([(1, i) for i in range(1, 2 * len(ks_state) + 1)])
    sim.plot_dispersion_relation()

    log.info(f"f cutoff = {cfg.omega_cutoff / 2 / np.pi}")

    S_matrix = sim.get_s_matrix().array
    _ = check_photon_conservation(S_matrix, cfg.omegas, cfg.omega_pump, cfg.ks_state)

    T_grid = sim._get_T_grid()
    _, _ = check_transfer_matrix_determinant(T_grid, tolerance=1e-12)

    # symbolic_matrix, _ = sim.get_symbolic_matrix()
    # a = CellSingleMode()
    # a.export_matrix_graphic(symbolic_matrix)

    plt.show()


if __name__ == "__main__":
    setup_logging()

    julia_comparison()
