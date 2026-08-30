import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from analysis.checks import check_photon_flux_conservation
from logger import get_logger
from models import JTLDiscrete
from simulation import Simulation, SimulationConfig

log = get_logger(__name__)


def track_gap_center_over_pump_frequency():
    pump_frequencies = np.linspace(1.5, 12, 12) * 2 * np.pi

    ks_state = [0, 1]
    ncell = 320

    gap_min = np.zeros(len(pump_frequencies))

    freq_min = 1  # GHz
    freq_max = 12  # GHz
    n_freqs = 1000
    signal_freqs = np.linspace(freq_min, freq_max, n_freqs) * 2 * np.pi

    window_ghz = 1.0
    df = (freq_max - freq_min) / (n_freqs - 1)  # GHz per point
    window_half = int(window_ghz / df)
    n_window = 2 * window_half
    freq_offset_axis = np.arange(-window_half, window_half) * df  # GHz
    phot_cons_window = np.full((len(pump_frequencies), n_window), np.nan)

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
            epsilon_nramp=0,
        )

        sim = Simulation(JTLDiscrete, cfg)
        smat = sim.get_s_matrix(normalize=True)
        gap_min_index = np.abs(smat.array)[:, 1, 3].argmin()
        gap_min[index] = signal_freqs[gap_min_index] / 2 / np.pi

        check = check_photon_flux_conservation(smat.array, signal_freqs, w_p, ks_state)

        # Extract window around gap_min_index, port 0 (signal input L)
        w_start = max(0, gap_min_index - window_half)
        w_stop = min(n_freqs, gap_min_index + window_half)
        i_out_start = window_half - (gap_min_index - w_start)
        i_out_stop = i_out_start + (w_stop - w_start)
        phot_cons_window[index, i_out_start:i_out_stop] = check[w_start:w_stop, 0]

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

    x_ghz = pump_frequencies / 2 / np.pi
    for i in range(len(x_ghz) - 1):
        x_pair = x_ghz[i : i + 2]
        y_pair = gap_min[i : i + 2]
        slope, intercept = np.polyfit(x_pair, y_pair, 1)
        ax.plot(
            x_pair,
            slope * x_pair + intercept,
            color="firebrick",
            linewidth=1.5,
            linestyle="--",
            label="Local slope" if i == 0 else None,
        )
        ax.annotate(
            f"{slope:.3f}",
            ((x_pair[0] + x_pair[1]) / 2, (y_pair[0] + y_pair[1]) / 2),
            textcoords="offset points",
            xytext=(0, 6),
            fontsize=10,
            color="firebrick",
            ha="center",
        )
    ax.legend()

    ax.set_xlabel("Pump Frequency (GHz)")
    ax.set_ylabel("Gap Center Frequency (GHz)")
    ax.set_title("Gap Center vs. Pump Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    im = ax2.pcolormesh(
        pump_frequencies / 2 / np.pi,
        freq_offset_axis,
        phot_cons_window.T,
        cmap="RdBu_r",
        vmin=0,
        vmax=2,
        shading="auto",
    )
    fig2.colorbar(
        im, ax=ax2, label=r"$\sum_i\,(\omega_i/\omega_j)\,|S_{ij}|^2$  (port 0)"
    )
    ax2.axhline(0, color="k", linestyle="--", linewidth=1, label="gap center")
    ax2.set_xlabel("Pump Frequency (GHz)")
    ax2.set_ylabel("Frequency offset from gap center (GHz)")
    ax2.set_title("Photon conservation around gap center vs. pump frequency")
    ax2.legend()
    fig2.tight_layout()

    # save_all("track_gap_center")
    plt.show()

