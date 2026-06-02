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
        omega_cutoff=2 * 50 / 530e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,  # usually smaller
        epsilon=0.055,  # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
        omega_c=3.4 * 2 * np.pi, # gap: well in the S parameter from signal to tranmission
        v_ratio=2.5,
        freq_min=1, # GHz
        freq_max=12, # GHz
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

def track_gap_center_over_pump_frequency():
    pump_frequencies = np.linspace(2, 14, 10) * 2 * np.pi
    
    ks_state = [0, 1]
    ncell = 320
    
    gap_min = np.zeros(len(pump_frequencies))
    
    freq_min=1 # GHz
    freq_max=12 # GHz
    n_freqs=500
    signal_freqs = np.linspace(freq_min, freq_max, n_freqs)

    for index, w_p in enumerate(pump_frequencies):
        cfg = SimulationConfig(
            Z0=50,
            M=1,
            ks_state=ks_state,
            ncell=ncell,
            cell_size=10e-6,
            omega_cutoff=2 * 50 / 530e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
            omega_pump=w_p,
            omega_j=35 * 2 * np.pi,  # usually smaller
            epsilon=0.055,  # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
            omega_c=3.4 * 2 * np.pi, # gap: well in the S parameter from signal to tranmission
            v_ratio=2.5,
            freq_min=freq_min, # GHz
            freq_max=freq_max, # GHz
            n_freqs=n_freqs,
            disorder=False,
            nramp=0,
        )

        sim = Simulation(JTL, cfg)
        gap_min_index = np.abs(sim.get_s_matrix().array)[:, 2, 0].argmin()
        # sim.plot_s_parameters([(i, 1) for i in range(1, 2 * len(ks_state) + 1)])
        # plt.show()
        gap_min[index] = signal_freqs[gap_min_index] / 2 / np.pi

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pump_frequencies / 2 / np.pi, gap_min, color="steelblue", linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Pump Frequency (GHz)", fontsize=13)
    ax.set_ylabel("Gap Center Frequency (GHz)", fontsize=13)
    ax.set_title("Gap Center vs. Pump Frequency", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    plt.show()

    return 


if __name__ == "__main__":
    setup_logging()

    julia_comparison()
