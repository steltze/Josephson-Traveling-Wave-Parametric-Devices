import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from simulation import SimulationConfig, Simulation
from models.jtl_discrete_multipump import JTLDiscreteMultiPump
from models.electrical_elements import multipump_frequency_grid
from examples.utils import save_all
from logger import get_logger, setup_logging

log = get_logger(__name__)

def two_pump_jtl(cell_topology: str = "pi", backend=None):
    """
    One JTL modulated by two independent pump tones at once (see
    models.jtl_discrete_multipump.JTLDiscreteMultiPump), each doing its
    own signal -> idler conversion:

        idler1 = signal + pump1
        idler2 = signal + pump2

    Kmax tracks each pump's sideband count (see
    models.electrical_elements.ModulatedInductorMultiPump), giving a
    prod(2*Kmax+1)-state tensor lattice.

    Uses the `Simulation` class with `cell_cls=JTLDiscreteMultiPump`.
    `Simulation._get_T_grid` only calls `cell_cls.build` +
    `single_mode_matrix_grid`, so it's agnostic to whether the lattice is
    the single-pump flat `ks_state` or the multi-pump tensor lattice.
    `get_s_matrix`'s photon-flux normalization is now multi-pump-aware too
    (branches on `cfg.omega_pump` being a list, building the port
    frequency grid via `multipump_frequency_grid` instead of `ks_state`).
    """
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ncell=321,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,
        omega_j=60 * 2 * np.pi,
        omega_pump = [6.8 * 2 * np.pi, 5 * 2 * np.pi],
        epsilon=[0.05, 0.03],
        v_ratio=[-2.5, -2.00],
        Kmax=[1, 1],
        freq_min=1,
        freq_max=14,
        n_freqs=500,
    )
    log.info(
        "pump frequencies: %.3f GHz, %.3f GHz",
        cfg.omega_pump[0] / (2 * np.pi),
        cfg.omega_pump[1] / (2 * np.pi),
    )

    sim = Simulation(JTLDiscreteMultiPump, cfg, backend=backend, cell_topology=cell_topology)
    S_total = sim.get_s_matrix(normalize=True).array  # (Nf, N, N)

    # Flat lattice index of each tracked (k1, k2) state, in the same
    # left-port order the S-matrix uses; the transmitted (right) port for
    # state index p sits at D + p.
    _, labels = multipump_frequency_grid(0.0, cfg.omega_pump, cfg.Kmax)
    D = len(labels)
    signal_idx = labels.index((0, 0))
    idler1_idx = labels.index((1, 0))
    idler2_idx = labels.index((0, 1))
    idler3_idx = labels.index((1, 1))
    
    freqs = cfg.freqs
    S31_signal = np.abs(S_total[:, D + signal_idx, signal_idx]) ** 2
    S_idler1 = np.abs(S_total[:, idler1_idx, signal_idx]) ** 2
    S_idler2 = np.abs(S_total[:, idler2_idx, signal_idx]) ** 2
    S_idler3 = np.abs(S_total[:, idler3_idx, signal_idx]) ** 2

    fp1 = cfg.omega_pump[0] / (2 * np.pi)
    fp2 = cfg.omega_pump[1] / (2 * np.pi)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs, 10 * np.log10(S31_signal + 1e-15), label=r"signal $\rightarrow$ signal")
    ax.plot(
        freqs,
        10 * np.log10(S_idler1 + 1e-15),
        label=rf"signal $\rightarrow$ idler$_1$ ($\omega_s+\omega_{{p1}}$, $f_{{p1}}={fp1:.2f}$ GHz)",
    )
    ax.plot(
        freqs,
        10 * np.log10(S_idler2 + 1e-15),
        label=rf"signal $\rightarrow$ idler$_2$ ($\omega_s+\omega_{{p2}}$, $f_{{p2}}={fp2:.2f}$ GHz)",
    )
    ax.plot(
        freqs,
        10 * np.log10(S_idler3 + 1e-15),
        label=rf"signal $\rightarrow$ idler$_3$ ($\omega_s+\omega_{{p1}}+\omega_{{p2}}$, $f_{{p1}}+f_{{p2}}={fp1 + fp2:.2f}$ GHz)",
    )
    ax.set_xlabel("Signal frequency (GHz)")
    ax.set_ylabel(r"$|S|^2$ (dB)")
    ax.set_title(f"Two-pump JTL — {cell_topology} cell, N={cfg.ncell}, ε={cfg.epsilon}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # save_all(prefix="multi_pump_jtl_example", fmt="svg")
    plt.show()


if __name__ == "__main__":
    setup_logging()
    two_pump_jtl()
