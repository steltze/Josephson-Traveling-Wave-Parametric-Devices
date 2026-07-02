import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete, JTLContinuous
from examples.utils import save_all

log = get_logger(__name__)


def compare_gap_bandwidth():
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0, 1],
        ncell=320,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 530e-3,
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.055,
        omega_c=3.4 * 2 * np.pi,
        v_ratio=2.5,
        freq_min=1,
        freq_max=12,
        n_freqs=500,
        disorder=False,
        nramp=0,
    )

    # --- discrete: measure bandwidth at -3 dB (|S31|² = 0.5) ---
    sim = Simulation(JTLDiscrete, cfg)
    S = sim.get_s_matrix().array
    S31_sq = np.abs(S[:, 2, 0]) ** 2
    freqs = cfg.freqs  # GHz
    omegas = cfg.omegas

    idx_min = np.argmin(S31_sq)
    threshold = 0.5  # -3 dB from unity baseline
    below = S31_sq < threshold
    edges = np.diff(below.astype(int))
    fall = np.where(edges == 1)[0]
    rise = np.where(edges == -1)[0]

    if len(fall) > 0 and len(rise) > 0:
        f_low_disc = freqs[fall[0]]
        f_high_disc = freqs[rise[0]]
        bw_disc_ghz = f_high_disc - f_low_disc
    else:
        f_low_disc = f_high_disc = bw_disc_ghz = float("nan")

    # --- analytical: Δω = 4γ / (dkS/dω + dkI/dω) ---
    sim_cont = JTLContinuous(cfg)
    omega_gap, delta_omega = sim_cont.gap_bandwidth()
    bw_cont_ghz = delta_omega / (2 * np.pi) if delta_omega is not None else float("nan")
    f_gap_cont = omega_gap / (2 * np.pi) if omega_gap is not None else float("nan")

    log.info(
        "Gap centre  — analytical: %.3f GHz   discrete: %.3f GHz",
        f_gap_cont,
        freqs[idx_min],
    )
    log.info(
        "Bandwidth   — analytical: %.3f GHz   discrete (-3 dB): %.3f GHz",
        bw_cont_ghz,
        bw_disc_ghz,
    )

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, 20 * np.log10(np.sqrt(S31_sq) + 1e-15), label="Discrete |S31|")
    ax.axhline(
        20 * np.log10(np.sqrt(threshold)),
        color="gray",
        ls="--",
        lw=0.8,
        label=f"-3 dB threshold",
    )
    ax.axvline(
        f_low_disc,
        color="C1",
        ls=":",
        lw=1.2,
        label=f"Discrete edges ({bw_disc_ghz:.3f} GHz)",
    )
    ax.axvline(f_high_disc, color="C1", ls=":", lw=1.2)
    ax.axvline(
        f_gap_cont - bw_cont_ghz / 2,
        color="C2",
        ls="--",
        lw=1.2,
        label=f"Analytical edges ({bw_cont_ghz:.3f} GHz)",
    )
    ax.axvline(f_gap_cont + bw_cont_ghz / 2, color="C2", ls="--", lw=1.2)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S31| (dB)")
    ax.set_title("Gap bandwidth: continuous model vs discrete")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # save_all("compare_gap_bandwidth")
    plt.show()

