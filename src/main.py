import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging

log = get_logger(__name__)
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete, jtl_continuous
from symbolic import CellSingleMode
from analysis.checks import (
    check_photon_conservation,
    check_energy_conservation,
    check_pump_photon_balance,
    check_transfer_matrix_determinant,
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

    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi * 1e9))

    # sim = Simulation(JTLDiscrete.left_handed(), cfg)
    sim = Simulation(JTLDiscrete, cfg)

    # symbolic_matrix, _ = sim.get_symbolic_matrix()
    # a = CellSingleMode()
    # a.export_matrix_graphic(symbolic_matrix)
    sim.plot_s_parameters([(1, 1), (2, 1), (3, 1), (4, 1)])
    sim.plot_s_parameters([(1, 1), (1, 2), (1, 3), (1, 4)])
    sim.plot_dispersion_relation()
    plt.show()


def julia_comparison():
    ks_state = [0, 1]
    ncell = 321
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=ks_state,
        ncell=ncell,
        cell_size=25e-6,
        omega_cutoff=2 * 50 / 540e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,  # usually smaller
        epsilon=0.055,  # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
        omega_c=3.4
        * 2
        * np.pi,  # gap: well in the S parameter from signal to tranmission
        v_ratio=2.5,
        freq_min=1,  # GHz
        freq_max=12,  # GHz
        n_freqs=1000,
        disorder=False,
        nramp=0,
    )

    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi))

    sim = Simulation(JTLDiscrete, cfg)
    sim.plot_s_parameters([(i, 1) for i in range(1, 2 * len(ks_state) + 1)])
    sim.plot_s_parameters([(1, i) for i in range(1, 2 * len(ks_state) + 1)])
    sim.plot_dispersion_relation()

    log.info(f"f cutoff = {cfg.omega_cutoff / 2 / np.pi}")

    S_matrix = sim.get_s_matrix().array
    T_grid = sim._get_T_grid()

    _, _ = check_transfer_matrix_determinant(T_grid, tolerance=1e-12)

    _, axes_phot = plt.subplots(1, 2, figsize=(12, 4))
    check_photon_conservation(
        S_matrix, cfg.omegas, cfg.omega_pump, cfg.ks_state, ax=axes_phot[0]
    )
    check_energy_conservation(
        S_matrix, cfg.omegas, cfg.omega_pump, cfg.ks_state, ax=axes_phot[1]
    )
    axes_phot[0].set_title("Photon check  (= 1 only for passive)")
    plt.tight_layout()

    # check_pump_photon_balance(S_matrix, cfg.omegas, cfg.omega_pump, cfg.ks_state)

    # symbolic_matrix, _ = sim.get_symbolic_matrix()
    # a = CellSingleMode()
    # a.export_matrix_graphic(symbolic_matrix)

    plt.show()


def harmonics_comparison():
    ncell = 320
    freq_min = 1  # GHz
    freq_max = 12  # GHz
    n_freqs = 1000
    signal_freqs = np.linspace(freq_min, freq_max, n_freqs) * 2 * np.pi

    plt.figure()

    db = np.zeros((2, len(signal_freqs)))
    energy_checks = np.zeros((2, len(signal_freqs)))

    for index, (M, ks_state, (i, j)) in enumerate(
        [(1, [0, 1], (2, 0)), (4, [-2, -1, 0, 1, 2], (7, 2))]
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
            omega_c=3.4
            * 2
            * np.pi,  # gap: well in the S parameter from signal to tranmission
            v_ratio=2.5,
            freq_min=freq_min,  # GHz
            freq_max=freq_max,  # GHz
            n_freqs=n_freqs,
            disorder=False,
            nramp=0,
        )

        sim = Simulation(JTLDiscrete, cfg)
        T_grid = sim._get_T_grid()
        _, _ = check_transfer_matrix_determinant(T_grid, tolerance=1e-12)

        S_matrix = sim.get_s_matrix().array
        # _ = check_photon_conservation(S_matrix, cfg.omegas, cfg.omega_pump, cfg.ks_state)
        energy_checks[index] = check_energy_conservation(
            S_matrix, cfg.omegas, cfg.omega_pump, cfg.ks_state
        )[:, j]

        db[index] = 20.0 * np.log10(np.abs(S_matrix[:, i, j]))

    plt.figure()
    plt.plot(signal_freqs / 2 / np.pi, db[0], label="M=1")
    plt.plot(signal_freqs / 2 / np.pi, db[1], label="M=3")
    plt.legend()

    plt.figure()
    plt.plot(signal_freqs / 2 / np.pi, energy_checks[0], label="M=1")
    plt.plot(signal_freqs / 2 / np.pi, energy_checks[1], label="M=3")
    plt.legend()

    print(np.abs(db[1] - db[0]).sum())
    print(np.abs(energy_checks[1] - energy_checks[0]).sum())

    # symbolic_matrix, _ = sim.get_symbolic_matrix()
    # a = CellSingleMode()
    # a.export_matrix_graphic(symbolic_matrix)

    plt.show()


def track_gap_center_over_pump_frequency():
    pump_frequencies = np.linspace(2, 14, 10) * 2 * np.pi

    ks_state = [0, 1]
    ncell = 320

    gap_min = np.zeros(len(pump_frequencies))

    freq_min = 1  # GHz
    freq_max = 12  # GHz
    n_freqs = 500
    signal_freqs = np.linspace(freq_min, freq_max, n_freqs) * 2 * np.pi

    for index, w_p in enumerate(pump_frequencies):
        cfg = SimulationConfig(
            Z0=50,
            M=1,
            ks_state=ks_state,
            ncell=ncell,
            cell_size=10e-6,
            omega_cutoff=2 * 50 / 530e-3,  # L = 530 pH, C = 212 fF -> ~30 GHz
            omega_pump=w_p,
            omega_j=60 * 2 * np.pi,  # usually smaller
            epsilon=0.055,  # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
            omega_c=3.4
            * 2
            * np.pi,  # gap: well in the S parameter from signal to tranmission
            v_ratio=2.5,
            freq_min=freq_min,  # GHz
            freq_max=freq_max,  # GHz
            n_freqs=n_freqs,
            disorder=False,
            nramp=0,
        )

        sim = Simulation(JTLDiscrete, cfg)
        gap_min_index = np.abs(sim.get_s_matrix().array)[:, 2, 0].argmin()
        gap_min[index] = signal_freqs[gap_min_index] / 2 / np.pi

    fig, ax = plt.subplots(figsize=(8, 5))
    log.debug(
        f"Slope = {(gap_min[1] - gap_min[0]) / (pump_frequencies[1] / 2 / np.pi - pump_frequencies[0] / 2 / np.pi)}"
    )
    ax.plot(
        pump_frequencies / 2 / np.pi,
        gap_min,
        color="steelblue",
        linewidth=2,
        marker="o",
        markersize=5,
    )
    ax.set_xlabel("Pump Frequency (GHz)", fontsize=13)
    ax.set_ylabel("Gap Center Frequency (GHz)", fontsize=13)
    ax.set_title("Gap Center vs. Pump Frequency", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    plt.show()

    return


def continuous_vs_discrete():

    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0, 1],
        ncell=320,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 530e-3,  # 30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.055,
        omega_c=3.4 * 2 * np.pi,
        v_ratio=2.5,
        freq_min=1,
        freq_max=12,
        n_freqs=800,
        disorder=False,
        nramp=0,
    )

    freqs_ghz = cfg.freqs
    omegas = cfg.omegas
    omega_I = omegas + cfg.omega_pump

    # --- discrete ---
    sim = Simulation(JTLDiscrete, cfg)
    S = sim.get_s_matrix().array  # (Nf, 4, 4), 0-indexed
    S31_disc = (
        np.abs(S[:, 2, 0]) ** 2
    )  # signal transmission (right out, signal in left)
    S21_disc = (
        np.abs(S[:, 1, 0]) ** 2
    )  # backward idler (idler-left out, signal-left in)
    S11_disc = np.abs(S[:, 0, 0]) ** 2
    S41_disc = np.abs(S[:, 3, 0]) ** 2

    # print(S11_disc.max(), " --- ", ((omega_I / omegas) * S41_disc).max())

    # Energy check: (ωI/ωs)·|S21| + |S31| should ≈ 1 with pump
    energy_disc = (
        (omegas / omega_I) * S21_disc
        + S31_disc
        + S11_disc
        + (omegas / omega_I) * S41_disc
    )

    # --- continuous ---
    S31_cont, S21_cont = jtl_continuous.s_params(
        omegas,
        cfg.omega_pump,
        cfg.omega_cutoff,
        cfg.omega_j,
        cfg.epsilon,
        cfg.ncell,
        cfg.v_pump,
        cfg.cell_size,
    )

    S31_cont = jtl_continuous.manual_solution(
        omegas=omegas,
        omega_pump=cfg.omega_pump,
        omega_cutoff=cfg.omega_cutoff,
        omega_j=cfg.omega_j,
        epsilon=cfg.epsilon,
        ncell=cfg.ncell,
        cell_size=cfg.cell_size,
        v_ratio=2.5,
    )

    delta_k = jtl_continuous.phase_mismatch(
        omegas, cfg.omega_pump, cfg.omega_cutoff, cfg.v_pump, cfg.cell_size
    )
    gap_freq = jtl_continuous.gap_center(
        cfg.omega_pump, cfg.omega_cutoff, cfg.v_pump, cfg.cell_size
    )

    log.info(
        "Analytical gap centre: %.3f GHz   (discrete gap: find min of |S31|)",
        gap_freq / (2 * np.pi) if gap_freq is not None else float("nan"),
    )

    energy_cont = (omegas / omega_I) * S21_cont + S31_cont

    # --- 2×2 comparison plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(
        f"Discrete vs Continuous  |  N={cfg.ncell}, ε={cfg.epsilon}, "
        f"fp={cfg.omega_pump / (2 * np.pi):.2f} GHz",
        fontsize=12,
    )
    ax_s31, ax_s21, ax_dk, ax_en = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def _vline(ax):
        if gap_freq is not None:
            ax.axvline(
                gap_freq / (2 * np.pi),
                color="gray",
                lw=0.8,
                ls=":",
                label="gap (cont.)",
            )

    # S31 — signal transmission
    ax_s31.plot(freqs_ghz, 20 * np.log10(np.sqrt(S31_disc) + 1e-15), label="Discrete")
    ax_s31.plot(
        freqs_ghz,
        20 * np.log10(np.sqrt(np.abs(S31_cont)) + 1e-15),
        "--",
        label="Continuous",
    )
    _vline(ax_s31)
    ax_s31.set_ylabel("|S31| (dB)")
    ax_s31.set_title("Signal transmission S31")
    ax_s31.legend(fontsize=9)
    ax_s31.grid(True, alpha=0.3)

    # S21 — backward idler (main gap channel)
    ax_s21.plot(freqs_ghz, 20 * np.log10(np.sqrt(S21_disc) + 1e-15), label="Discrete")
    ax_s21.plot(
        freqs_ghz,
        20 * np.log10(np.sqrt(np.abs(S21_cont)) + 1e-15),
        "--",
        label="Continuous",
    )
    _vline(ax_s21)
    ax_s21.set_ylabel("|S21| (dB)")
    ax_s21.set_title("Backward idler S21  (main stop-band channel)")
    ax_s21.legend(fontsize=9)
    ax_s21.grid(True, alpha=0.3)

    # Phase mismatch
    ax_dk.plot(freqs_ghz, delta_k, lw=1.5, label="Δk = ks + kI − kp")
    ax_dk.axhline(0, color="k", lw=0.8, ls="--")
    _vline(ax_dk)
    ax_dk.set_xlabel("Frequency (GHz)")
    ax_dk.set_ylabel("Δk (rad/cell)")
    ax_dk.set_title("Phase mismatch  (backward-wave: ks + kI = kp at gap)")
    ax_dk.legend(fontsize=9)
    ax_dk.grid(True, alpha=0.3)

    # Energy conservation: (ωI/ωs)|S21| + |S31| ≈ 1
    ax_en.plot(freqs_ghz, energy_disc, label="Discrete")
    ax_en.plot(freqs_ghz, np.abs(energy_cont), "--", label="Continuous")
    ax_en.axhline(1.0, color="k", lw=0.8, ls="--", label="ideal = 1")
    _vline(ax_en)
    ax_en.set_xlabel("Frequency (GHz)")
    ax_en.set_ylabel("(ωI/ωs)·|S21|² + |S31|²")
    ax_en.set_title("Energy conservation check  (= 1 with pump)")
    ax_en.legend(fontsize=9)
    ax_en.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    setup_logging()

    # julia_comparison()

    # track_gap_center_over_pump_frequency()

    continuous_vs_discrete()
