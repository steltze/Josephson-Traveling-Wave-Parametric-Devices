import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete
from analysis.checks import (
    check_transfer_matrix_determinant,
    check_photon_flux_conservation,
)
from dashboard import Dashboard

log = get_logger(__name__)


def julia_comparison(dashboard):
    ks_state = [0, 1]
    M = 1
    ncell = 320
    cfg = SimulationConfig(
        Z0=50,
        M=M,
        ks_state=ks_state,
        ncell=ncell,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 530e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,  # usually smaller
        epsilon=0.04,
        phi_dc_frac=1/3,  # Phi_dc/Phi0, matches julia/josephsoncircuits_comparison.jl's Phi_dc_frac
        phi_rf_frac=0.01,
        omega_c=3.4 * 2 * np.pi,  # gap: well in the S parameter from signal to tranmission
        v_ratio=-2.5,  # > 0 => co-propagating, < 0 => counter-propagating
        freq_min=1,  # GHz
        freq_max=12,  # GHz
        n_freqs=500,
        disorder=False,
        epsilon_nramp=0,  # where the peak will be
        adiabatic_pump=False,
    )

    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi))
    log.info("omega_cutoff = %.3f GHz", cfg.omega_cutoff / (2 * np.pi))

    sim = Simulation(JTLDiscrete, cfg, backend="numpy", cell_topology="pi")
    S_params = sim.get_s_matrix(normalize=True).array

    T_grid = sim.get_transfer_matrix_grid()

    _, _ = check_transfer_matrix_determinant(T_grid, tolerance=1e-10)

    sim.plot_dispersion_relation()

    flux_conservation_S = check_photon_flux_conservation(
        S_params, cfg.omegas, cfg.omega_pump, cfg.ks_state
    )
    _, ax = plt.subplots()

    signal_left_index = ks_state.index(0)

    ax.plot(cfg.freqs, flux_conservation_S[:, signal_left_index])
    ax.axhline(1, color="gray", ls="--", lw=1)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"$\sum_i \eta_i |S_{i,0}|^2$")
    ax.set_title("Photon-flux conservation of port: signal left")
    ax.grid(True, alpha=0.25)

    plt.show()

    if dashboard:
        dashboard_runs = [S_params]
        dashboard_labels = ["TWPC"]
        Dashboard(
            dashboard_runs,
            freqs=cfg.freqs,
            labels=dashboard_labels,
            ks_state=cfg.ks_state,
        ).run()
    else:
        _, ax = plt.subplots()

        signal_left_index = ks_state.index(0)
        signal_right_index = len(ks_state) + signal_left_index

        S21 = S_params[:, signal_right_index, signal_left_index]  # port1 -> port2, direct
        S12 = S_params[:, signal_left_index+1, signal_left_index]  # port2 -> port1, direct

        ax.plot(cfg.freqs, 20 * np.log10(np.abs(S21)), label="S21")
        ax.plot(cfg.freqs, 20 * np.log10(np.abs(S12)), label="S12")
        ax.set_xlabel("Signal Frequency (GHz)")
        ax.set_ylabel("|S| (dB)")
        ax.legend()
        plt.show()



if __name__ == "__main__":
    setup_logging()
    julia_comparison(dashboard=False)
