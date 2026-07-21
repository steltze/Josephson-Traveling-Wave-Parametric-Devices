import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete
from analysis.checks import check_transfer_matrix_determinant, check_photon_flux_conservation
from examples.utils import save_all

log = get_logger(__name__)


def julia_comparison():
    ks_state = [0, 1]
    M = 1
    ncell = 320+1
    cfg = SimulationConfig(
        Z0=50,
        M=M,
        ks_state=ks_state,
        ncell=ncell,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=30 * 2 * np.pi,  # usually smaller
        epsilon=0.045,  # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
        omega_c=3.4
        * 2
        * np.pi,  # gap: well in the S parameter from signal to tranmission
        v_ratio=-2.5, # > 0 => co-propagating, < 0 => counter-propagating
        freq_min=1,  # GHz
        freq_max=14,  # GHz
        n_freqs=1000,
        disorder=False,
        nramp=0,
    )
    central_band = 1
    transmitted_band = 3 # central_band + len(cfg.ks_state) 
    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi))

    sim = Simulation(JTLDiscrete, cfg)
    sim.plot_s_parameters([(transmitted_band, central_band), (central_band, transmitted_band)], k=0, normalize=True)

    # # sim.plot_s_parameters([(i, central_band) for i in range(1, 2 * len(ks_state) + 1)])
    # # sim.plot_s_parameters([(i, transmitted_band) for i in range(1, 2 * len(ks_state) + 1)])

    # log.info(f"f cutoff = {cfg.omega_cutoff / 2 / np.pi}")

    # T_grid = sim._get_T_grid()

    # _, _ = check_transfer_matrix_determinant(T_grid, tolerance=1e-18)

    # S_ph = sim.get_s_matrix(normalize=True).array
    # flux_check = check_photon_flux_conservation(
    #     S_ph, cfg.omegas, cfg.omega_pump, cfg.ks_state
    # )

    # fig, ax = plt.subplots()
    # ax.plot(cfg.freqs, flux_check[:, central_band-1], label="Input port 1 (signal)")
    # ax.axhline(1.0, color="k", lw=0.8, ls="--", label="ideal = 1")
    # ax.set_xlabel("Frequency (GHz)")
    # ax.set_ylabel(r"$\sum_i\,\eta_i\,|S_{ij}|^2$")
    # ax.set_title("Photon flux conservation")
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    # symbolic_matrix, _ = sim.get_symbolic_matrix()
    # a = CellSingleMode()
    # a.export_matrix_graphic(symbolic_matrix)

    # save_all(prefix="julia_comp", fmt="svg")
    epsilon_max = 0.2
    epsilon_values = np.linspace(0.03, epsilon_max, 2)
    colors = ["blue", "red"]
    linestyles = ["-", "--"]
    ncell_values = np.linspace(400, 1500, 2, dtype=int)


    fig, ax2 = plt.subplots()
    for i, ncell_n in enumerate(epsilon_values):
        cfg_eps = replace(cfg, epsilon=ncell_n)
        sim_eps = Simulation(JTLDiscrete, cfg_eps, backend="numpy")
        S_ph_eps = sim_eps.get_s_matrix(normalize=True).array

        T_grid = sim_eps._get_T_grid()

        _, _ = check_transfer_matrix_determinant(T_grid, tolerance=1e-18)

        # for i in range(2*len(ks_state)):
        for j in [0, 1]:
            ax2.plot(
                cfg.freqs,
                (10*np.log10(np.abs(S_ph_eps) ** 2))[:, j, 0],
                label=f"S{j+1}1, e={ncell_n:.2f}",
                color=colors[i],
                linestyle=linestyles[j % len(linestyles)],
            )

    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel(r"$|S_{i," + str(central_band) + r"}|^2$ (dB)")
    ax2.set_title("Signal transmission vs. modulation depth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.show()

