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
from dashboard import Dashboard

log = get_logger(__name__)


def julia_comparison(dashboard=False):
    ks_state = [0, 1]
    M = 1
    ncell = 2000
    cfg = SimulationConfig(
        Z0=50,
        M=M,
        ks_state=ks_state,
        ncell=ncell,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,  # usually smaller
        epsilon=0.06,  # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
        omega_c=3.4
        * 2
        * np.pi,  # gap: well in the S parameter from signal to tranmission
        v_ratio=-2.5, # > 0 => co-propagating, < 0 => counter-propagating
        freq_min=1,  # GHz
        freq_max=12,  # GHz
        n_freqs = 500,
        disorder=False,
        epsilon_nramp=0, # where the peak will be
        adiabatic_pump=True,
    )
    central_band = 1
    transmitted_band = 3 # central_band + len(cfg.ks_state) 
    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi))

    # sim = Simulation(JTLDiscrete, cfg)

    # epsilon_max = 0.2
    # epsilon_values = np.linspace(0.03, epsilon_max, 2)
    # ncell_values = np.linspace(400, 1500, 2, dtype=int)

    epsilon_ramp_values = [0, cfg.ncell//5, cfg.ncell//2]
    colors = ["blue", "red", "green"]
    linestyles = ["-", "--", "."]


    dashboard_runs = []
    dashboard_labels = []

    fig, ax2 = plt.subplots()
    for i, epsilon_ramp in enumerate(epsilon_ramp_values):
        cfg_eps = replace(cfg, epsilon_nramp=epsilon_ramp)
        sim_eps = Simulation(JTLDiscrete, cfg_eps, backend="numpy")
        S_ph_eps = sim_eps.get_s_matrix(normalize=True).array
        dashboard_runs.append(S_ph_eps)
        dashboard_labels.append(rf"$\epsilon_{{\mathrm{{envelope}}}}={epsilon_ramp:.3f}$")

        T_grid = sim_eps._get_T_grid()

        _, _ = check_transfer_matrix_determinant(T_grid)

        # for i in range(2*len(ks_state)):
        for j in [2]:
            ax2.plot(
                cfg.freqs,
                (10*np.log10(np.abs(S_ph_eps) ** 2))[:, j, 0],
                label=rf"$\epsilon_{{\mathrm{{envelope}}}}={epsilon_ramp}$",
                # color=colors[i],
                # linestyle=linestyles[j % len(linestyles)],
            )

    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel(r"$|S_{i," + str(central_band) + r"}|^2$ (dB)")
    ax2.set_title(f"Signal transmission vs. modulation depth, ncells = {cfg.ncell}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    sim_eps.plot_dispersion_relation()

    plt.show()

    if dashboard:
        Dashboard(dashboard_runs, freqs=cfg.freqs, labels=dashboard_labels, ks_state=cfg.ks_state).run()

