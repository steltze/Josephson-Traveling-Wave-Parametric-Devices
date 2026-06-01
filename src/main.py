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
    ks_state = [0, 1]
    ncell = 320
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=ks_state,
        ncell=ncell,
        cell_size=10e-6,
        omega_cutoff=2*50 / 530e-3,  # L = 530 pH, C = 212 fF → ~30 GHz
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi, # usually smaller
        epsilon=0.055,  # 10% inductance modulation (|Φ_RF| = 0.01 Φ_0)
        omega_c=3.4 * 2 * np.pi,
        v_ratio=2.5,
        freq_min=1,
        freq_max=14,
        n_freqs=1000,
        disorder=False,
        nramp=0,
    )

    logging.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi))

    sim = Simulation(JTL, cfg)

    sim.plot_s_parameters([(i, 1) for i in range(1, 2*len(ks_state)+1)])
    sim.plot_s_parameters([(1, i) for i in range(1, 2*len(ks_state)+1)])
    sim.plot_dispersion_relation()

    T_grid = sim._get_T_grid()

    idx = np.argmax(np.max(np.abs(T_grid), axis=(-3, -2,-1)))
    print((T_grid[idx, 10]))
    
    T_total = T_grid[:, 0].copy()
    for c in range(1, ncell):
        T_total = T_grid[:, c] @ T_total
    idx = np.argmax(np.max(np.abs(T_total), axis=(-2,-1)))
    print(np.abs(T_total[idx]))
    

    print("adawdawd", (np.abs(np.linalg.det(T_grid)-1.0)!=0.0).sum())

    logging.info(f"f cutoff = {cfg.omega_cutoff/2/np.pi}")

    omega_signals = cfg.omegas  # rad/s
    omega_idlers = omega_signals + cfg.omega_pump  # rad/s
    S_matrix = np.abs(sim.get_s_matrix().array)
    check1 = (
        S_matrix[:, 0, 0] ** 2
        + (omega_signals / omega_idlers) * S_matrix[:, 1, 0] ** 2
        + S_matrix[:, 2, 0] ** 2
        + (omega_signals / omega_idlers) * S_matrix[:, 3, 0] ** 2
    )
    check2 = (
        S_matrix[:, 1, 1] ** 2
        + (omega_idlers / omega_signals) * S_matrix[:, 0, 1] ** 2
        + S_matrix[:, 3, 1] ** 2
        + (omega_idlers / omega_signals) * S_matrix[:, 2, 1] ** 2
    )
    plt.figure()
    plt.plot(omega_signals / 2 / np.pi, check1, label="Signal")
    plt.plot(omega_signals / 2 / np.pi, check2, label="Idler")
    plt.legend()
    logging.info(omega_signals[check1.argmax()] / 2 / np.pi)

    from analysis.checks import plot_verification

    # mm = plot_verification(S_matrix, omega_signals, cfg.omega_pump, [0, 1])
    symbolic_matrix, _ = sim.get_symbolic_matrix()
    a = CellSingleMode()
    a.export_matrix_graphic(symbolic_matrix)
    plt.show()

def prepare_manual_ABCDs(Z0 = 50,
    ncell = 320,
    cell_size = 10e-6,
    omega_cutoff = 2 * 50 / 530e-12,  # L = 530 pH, C = 212 fF -> ~30 GHz
    omega_pump = 6.8e9 * 2 * np.pi,
    omega_j = 60e9 * 2 * np.pi,
    epsilon = 0.1,
    v_ratio = 2.5,
    freq_min = 1e9,
    freq_max = 12e9,
    n_freqs = 800):


    omegas = np.linspace(freq_min, freq_max, n_freqs) * 2 * np.pi
    ns = np.arange(ncell)

    ZR = Z0 * np.ones(ncell)
    L = ZR / omega_cutoff * 2
    C = 2 * 1.0 / (omega_cutoff * ZR)

    v_signal = cell_size * omega_cutoff / 2
    w_p = omega_pump
    v_p = v_signal / v_ratio
    thetas = w_p / v_p * ns * cell_size

    # Broadcast (n_freqs, 1) × (1, ncell) → (n_freqs, ncell)
    w0 = omegas[:, None]
    w1 = (omegas + w_p)[:, None]
    
    L_ = L[None, :]
    L_[0, 0] = 0.0

    C_ = C[None, :]
    C_[0, 0] = C_[0, 0]/2.0
    C_[0, -1] = C_[0, -1]/2.0

    denom0 = 1 - w0**2 / omega_j**2
    denom1 = 1 - w1**2 / omega_j**2

    Zs0_0 = 1j * w0 * L_ / denom0
    Zs0_1 = 1j * w1 * L_ / denom1
    Zs1_0 = 1j * w0 * L_ * epsilon / (2 * denom0**2)
    Zs1_1 = 1j * w1 * L_ * epsilon / (2 * denom1**2)
    Yg0_0 = 1j * w0 * C_
    Yg0_1 = 1j * w1 * C_

    exp_neg = np.exp(-1j * thetas)[None, :]
    exp_pos = np.exp(+1j * thetas)[None, :]

    ABCDs = np.zeros((n_freqs, ncell, 4, 4), dtype=complex)

    # Row 0
    ABCDs[:, :, 0, 0] = 1
    ABCDs[:, :, 0, 2] = -Zs0_0
    ABCDs[:, :, 0, 3] = -Zs1_1 * exp_neg
    # Row 1
    ABCDs[:, :, 1, 1] = 1
    ABCDs[:, :, 1, 2] = -Zs1_0 * exp_pos
    ABCDs[:, :, 1, 3] = -Zs0_1
    # Row 2  (Yg1 = 0 so those terms vanish)
    ABCDs[:, :, 2, 0] = -Yg0_0
    ABCDs[:, :, 2, 2] = Yg0_0 * Zs0_0 + 1
    ABCDs[:, :, 2, 3] = Yg0_1 * Zs1_1 * exp_neg
    # Row 3
    ABCDs[:, :, 3, 1] = -Yg0_1
    ABCDs[:, :, 3, 2] = Yg0_0 * Zs1_0 * exp_pos
    ABCDs[:, :, 3, 3] = Yg0_1 * Zs0_1 + 1

    return ABCDs  # (n_freqs, ncell, 4, 4)


def compare_abcd_vs_sim():
    """
    Compare per-cell T-matrices and cascaded S-matrices between the symbolic
    simulation path (Simulation/JTL) and the manual vectorised path
    (prepare_manual_ABCDs).

    Notable structural differences that will show up:
      - Cell 0: sim sets Zs0=0 (no series element); manual uses full Zs0.
      - Cells 0 and N-1: sim uses C/2 (symmetric pi-section); manual uses C.
    """
    from solver.s_matrix import ABCD_to_S

    Z0 = 50
    ncell = 320
    cell_size = 10e-6
    omega_cutoff = 2*50 / 530e-12
    omega_pump = 6.8e9 * 2 * np.pi
    omega_j = 60e9 * 2 * np.pi
    epsilon = 0.1
    omega_c = 3.4e9 * 2 * np.pi
    v_ratio = 2.5
    freq_min = 1e9
    freq_max = 12e9
    n_freqs = 800

    cfg = SimulationConfig(
        Z0=Z0, M=1, ks_state=[0, 1], ncell=ncell, cell_size=cell_size,
        omega_cutoff=omega_cutoff, omega_pump=omega_pump, omega_j=omega_j,
        epsilon=epsilon, omega_c=omega_c, v_ratio=v_ratio,
        freq_min=freq_min, freq_max=freq_max, n_freqs=n_freqs,
        disorder=False, nramp=0,
    )
    sim = Simulation(JTL, cfg)

    T_grid = sim._get_T_grid()      # (Nf, Nc, 4, 4) — sim per-cell T-matrices
    # T_grid = np.delete(T_grid, [0, 1, 4, 5, 6, 9], axis=2)
    # T_grid = np.delete(T_grid, [0, 1, 4, 5, 6, 9], axis=3)
    
    S_sim = sim.get_s_matrix().array  # (Nf, 4, 4) — sim total S
    # S_sim = np.delete(S_sim, [0, 1, 4, 5, 6, 9], axis=1)
    # S_sim = np.delete(S_sim, [0, 1, 4, 5, 6, 9], axis=2)
    T_manual = prepare_manual_ABCDs(Z0, ncell, cell_size, omega_cutoff, omega_pump, omega_j, epsilon, v_ratio, freq_min, freq_max, n_freqs)

    freqs_GHz = cfg.freqs / 1e9
    Nf, Nc = T_grid.shape[:2]

    # # --- 1. Per-cell T-matrix comparison (inner cells; end cells differ by design) ---
    inner = slice(1, Nc - 1)
    err = np.abs(T_grid[:, inner] - T_manual[:, inner])
    rel = err / (np.abs(T_grid[:, inner]) + 1e-30)

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
    axes1[0].semilogy(freqs_GHz, err.mean(axis=(1, 2, 3)), label='mean |ΔT|')
    axes1[0].semilogy(freqs_GHz, err.max(axis=(1, 2, 3)), label='max |ΔT|')
    axes1[0].set_xlabel('Frequency (GHz)')
    axes1[0].set_ylabel('Absolute error per element')
    axes1[0].set_title('T-matrix absolute error — inner cells')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.25)

    axes1[1].semilogy(freqs_GHz, rel.mean(axis=(1, 2, 3)), label='mean rel')
    axes1[1].semilogy(freqs_GHz, rel.max(axis=(1, 2, 3)), label='max rel')
    axes1[1].set_xlabel('Frequency (GHz)')
    axes1[1].set_ylabel('Relative error per element')
    axes1[1].set_title('T-matrix relative error — inner cells')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.25)
    fig1.suptitle('Per-cell T-matrix: sim vs prepare_manual_ABCDs')
    fig1.tight_layout()

    # --- 2. Total S-matrix comparison ---
    Q, R = np.linalg.qr(T_manual[:, 0])
    for c in range(1, Nc):
        Q_new, R = np.linalg.qr(R @ T_manual[:, c])
        Q = Q @ Q_new
    # T_total = Q @ R  →  T_total_inv = R^{-1} @ Q^H (triangular solve)
    S_man = ABCD_to_S(np.linalg.solve(R, Q.conj().swapaxes(-1, -2)), Z0)

    # plt.plot(freqs_GHz, 20.0 * np.log10(np.abs(S_man[:, :, 0])))

    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    port_pairs = [(0, 0), (1, 0), (2, 0), (3, 0)]  # S11, S21, S31, S41
    for col, (i, j) in enumerate(port_pairs):
        lbl = f'S{i+1}{j+1}'
        sim_dB = 20 * np.log10(np.abs(S_sim[:, i, j]) + 1e-30)
        man_dB = 20 * np.log10(np.abs(S_man[:, i, j]) + 1e-30)

        axes2[0, col].plot(freqs_GHz, sim_dB, label='sim')
        axes2[0, col].plot(freqs_GHz, man_dB, '--', label='manual')
        axes2[0, col].set_title(lbl)
        axes2[0, col].set_xlabel('Frequency (GHz)')
        axes2[0, col].set_ylabel('|S| (dB)')
        axes2[0, col].legend()
        axes2[0, col].grid(True, alpha=0.25)

        axes2[1, col].plot(freqs_GHz, sim_dB - man_dB)
        axes2[1, col].set_title(f'Δ{lbl}  (sim − manual, dB)')
        axes2[1, col].set_xlabel('Frequency (GHz)')
        axes2[1, col].set_ylabel('ΔdB')
        axes2[1, col].axhline(0, color='k', lw=0.5)
        axes2[1, col].grid(True, alpha=0.25)

    fig2.suptitle('S-matrix comparison: sim vs prepare_manual_ABCDs')
    fig2.tight_layout()

    logging.info("Max |ΔS|: %.4e   Mean |ΔS|: %.4e",
                 np.abs(S_sim - S_man).max(), np.abs(S_sim - S_man).mean())
    
    omega_signals = cfg.omegas  # rad/s
    omega_idlers = omega_signals + cfg.omega_pump  # rad/s
    S_matrix = np.abs(S_man)
    check1 = (
        S_matrix[:, 0, 0] ** 2
        + (omega_signals / omega_idlers) * S_matrix[:, 1, 0] ** 2
        + S_matrix[:, 2, 0] ** 2
        + (omega_signals / omega_idlers) * S_matrix[:, 3, 0] ** 2
    )
    check2 = (
        S_matrix[:, 1, 1] ** 2
        + (omega_idlers / omega_signals) * S_matrix[:, 0, 1] ** 2
        + S_matrix[:, 3, 1] ** 2
        + (omega_idlers / omega_signals) * S_matrix[:, 2, 1] ** 2
    )
    plt.figure()
    plt.plot(omega_signals, check1)
    plt.plot(omega_signals, check2)
    plt.show()


if __name__ == "__main__":
    main2()
