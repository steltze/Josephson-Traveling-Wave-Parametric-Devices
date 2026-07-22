import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete, JTLContinuous
from analysis.checks import check_photon_flux_conservation
from examples.utils import save_all

log = get_logger(__name__)


def _base_config(**overrides):
    cfg_kwargs = dict(
        Z0=50,
        M=1,
        ks_state=[-1, 0],
        ncell=321,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,
        omega_pump=13.31 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.04,
        omega_c=3.4 * 2 * np.pi,
        v_ratio=1.0,
        freq_min=1,
        freq_max=14,
        n_freqs=500,  # increase to 5000 to see interesting spikes
        disorder=False,
        nramp=0,
    )
    cfg_kwargs.update(overrides)
    return SimulationConfig(**cfg_kwargs)


def _discrete_vs_continuous(
    cfg, mode: str, idler_ks: int, label: str, save_name: str, backend=None
):
    """
    Compare the discrete Floquet solver against JTLContinuous's closed-form
    coupled-mode BVP for a given interaction ``mode``.

    ``idler_ks`` is the discrete sideband (a value from cfg.ks_state) that
    corresponds to the continuous model's idler for this mode:
        amplifier (ωI = ωs + ωp)  -> idler_ks = +1
        converter (ωI = ωp - ωs)  -> idler_ks = -1

    ``backend`` selects the numerical backend for the discrete cascade (see
    ``backends.available_backends()``). Defaults to "numpy". Note that for
    this config's port count (N=4, from ks_state=[-1, 0]), the "numba"
    backend measures ~2x *slower* than "numpy", even with the whole
    per-cell cascade fused into one compiled call (Backend.cascade_all) —
    the bottleneck for this config isn't Python-level dispatch, it's fixed
    per-call overhead in solving many tiny (2x2) complex linear systems.
    Numba pulls ahead once the port count grows (larger M / longer
    ks_state means bigger per-frequency solves), not for this config.
    """
    log.info(f"f cutoff = {cfg.omega_cutoff / 2 / np.pi}")

    freqs_ghz = cfg.freqs
    omegas = cfg.omegas

    # --- discrete ---
    sim = Simulation(JTLDiscrete, cfg, backend=backend)
    S = sim.get_s_matrix(normalize=True).array  # (Nf, 4, 4), 0-indexed
    central_band = cfg.ks_state.index(0)
    # idler_band = cfg.ks_state.index(idler_ks)
    transmitted_band = central_band + len(cfg.ks_state)
    S21_disc = np.abs(S[:, transmitted_band-1, central_band]) ** 2
    S31_disc = np.abs(S[:, transmitted_band, central_band]) ** 2

    # Ports with a negative signed frequency (idlers, ks<0) are pseudo-unitary
    # partners rather than ordinary channels — a plain power sum overestimates
    # gain there. Use the eta-weighted (Manley-Rowe) conservation law instead.
    energy_disc = check_photon_flux_conservation(
        S, omegas, cfg.omega_pump, cfg.ks_state
    )[:, central_band]

    # --- continuous ---
    sim_cont = JTLContinuous(cfg, mode=mode)
    S_cont = sim_cont.get_s_matrix().array
    S31_cont = np.abs(S_cont[:, 2, 0]) ** 2
    S21_cont = np.abs(S_cont[:, 1, 0]) ** 2

    delta_k = sim_cont.phase_mismatch(omegas)
    gap_freq, gap_bandwidth = sim_cont.gap_bandwidth()

    log.info(
        "[%s] Analytical gap centre: %.3f GHz   (discrete gap: find min of |S31|)",
        label,
        gap_freq / (2 * np.pi) if gap_freq is not None else float("nan"),
    )

    if mode == "converter":
        energy_cont = S21_cont + S31_cont
    else:
        energy_cont = S21_cont - S31_cont

    # --- 2×2 comparison plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(
        f"Discrete vs Continuous — {label}  |  N={cfg.ncell}, ε={cfg.epsilon}, "
        f"fp={cfg.omega_pump / (2 * np.pi):.2f} GHz",
    )
    ax_s31, ax_s21, ax_dk, ax_en = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def _vline(ax):
        if gap_freq is not None:
            if gap_bandwidth is not None:
                gap_lo = (gap_freq - gap_bandwidth / 2) / (2 * np.pi)
                gap_hi = (gap_freq + gap_bandwidth / 2) / (2 * np.pi)
                ax.axvspan(
                    gap_lo,
                    gap_hi,
                    color="gray",
                    alpha=0.15,
                    zorder=0,
                    label="gap width",
                )
            ax.axvline(
                gap_freq / (2 * np.pi),
                color="gray",
                lw=2.2,
                ls=":",
                zorder=1,
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
    ax_s31.set_ylabel(r"$|S_{s_L \rightarrow s_R}| (dB)$")
    ax_s31.set_title(r"Signal transmission $S_{s_L \rightarrow s_R}$")
    ax_s31.legend()
    ax_s31.grid(True, alpha=0.3)

    # S21 — idler (main gap channel)
    ax_s21.plot(freqs_ghz, 20 * np.log10(np.sqrt(S21_disc) + 1e-15), label="Discrete")
    ax_s21.plot(
        freqs_ghz,
        20 * np.log10(np.sqrt(np.abs(S21_cont)) + 1e-15),
        "--",
        label="Continuous",
    )
    _vline(ax_s21)
    ax_s21.set_ylabel(r"$|S_{s_L \rightarrow i_L}| (dB)$")
    ax_s21.set_title(r"Signal conversion $S_{s_L \rightarrow i_L}$")
    ax_s21.legend()
    ax_s21.grid(True, alpha=0.3)

    # Phase mismatch
    ax_dk.plot(freqs_ghz, delta_k, lw=1.5)
    ax_dk.axhline(0, color="k", lw=0.8, ls="--")
    _vline(ax_dk)
    ax_dk.set_xlabel("Frequency (GHz)")
    ax_dk.set_ylabel("Δk (rad/cell)")
    ax_dk.set_title("Phase mismatch = ks + kp - ki")
    ax_dk.legend()
    ax_dk.grid(True, alpha=0.3)

    # Energy conservation: (ωI/ωs)|S21| + |S31| ≈ 1
    ax_en.plot(freqs_ghz, energy_disc, label="Discrete")
    ax_en.plot(freqs_ghz, np.abs(energy_cont), "--", label="Continuous")
    ax_en.axhline(1.0, color="k", lw=0.8, ls="--", label="ideal = 1")
    _vline(ax_en)
    ax_en.set_xlabel("Frequency (GHz)")
    ax_en.set_ylabel("|S21|² + |S31|²")
    ax_en.set_title("Energy conservation check  (= 1 with pump)")
    ax_en.legend()
    ax_en.grid(True, alpha=0.3)

    plt.tight_layout()

    # --- error between discrete and continuous S31, in dB ---
    S31_disc_db = 20 * np.log10(np.sqrt(S31_disc) + 1e-15)
    S31_cont_db = 20 * np.log10(np.sqrt(np.abs(S31_cont)) + 1e-15)
    S31_err_db = S31_disc_db - S31_cont_db
    rms_err_db = np.sqrt(np.mean(S31_err_db**2))

    fig_err, ax_err = plt.subplots(figsize=(8, 5))
    # fig_err.suptitle(
    #     f"Discrete vs Continuous Model Error — {label}  |  N={cfg.ncell}, ε={cfg.epsilon}, "
    #     f"fp={cfg.omega_pump / (2 * np.pi):.2f} GHz",
    # )
    ax_err.axhline(0, color="0.4", lw=1.0, ls="-", zorder=1)
    ax_err.plot(
        freqs_ghz,
        S31_err_db,
        color="#C0392B",
        lw=1.6,
        label="Discrete - Continuous",
        zorder=3,
    )
    ax_err.fill_between(freqs_ghz, S31_err_db, 0, color="#C0392B", alpha=0.15, zorder=2)
    # ax_err.axhline(
    #     rms_err_db,
    #     color="0.3",
    #     lw=1.0,
    #     ls="--",
    #     label=f"RMS = {rms_err_db:.2f} dB",
    #     zorder=2,
    # )
    _vline(ax_err)
    ax_err.set_xlabel("Frequency (GHz)")
    ax_err.set_ylabel(r"$S_{s_L \rightarrow s_R}$ error (dB)")
    ax_err.set_title(r"Signal transmission $S_{s_L \rightarrow s_R}$: model discrepancy")
    ax_err.legend(loc="best", frameon=True)
    ax_err.grid(True, which="major", alpha=0.3)
    ax_err.minorticks_on()
    ax_err.grid(True, which="minor", alpha=0.1)
    fig_err.tight_layout()

    # save_all(prefix=save_name, fmt="svg")
    plt.show()


def continuous_vs_discrete_amplifier(backend=None):
    """
    Validate JTLContinuous's amplifier mode (ωI = ωs + ωp) against JTLDiscrete.

    backend : numerical backend for the discrete cascade, e.g. "numba"
        (see `backends.available_backends()`; default "numpy" — see the
        perf note on `_discrete_vs_continuous` before switching this).
    """
    cfg = _base_config()
    _discrete_vs_continuous(
        cfg,
        mode="amplifier",
        idler_ks=-1,
        label="Amplifier, ωI=ωs+ωp",
        save_name="continuous_vs_discrete_amplifier",
        backend=backend,
    )


def continuous_vs_discrete_converter(backend=None):
    """
    Validate JTLContinuous's converter mode (ωI = ωp - ωs) against JTLDiscrete.

    backend : numerical backend for the discrete cascade, e.g. "numba"
        (see `backends.available_backends()`; default "numpy" — see the
        perf note on `_discrete_vs_continuous` before switching this).
    """
    cfg = _base_config()
    _discrete_vs_continuous(
        cfg,
        mode="converter",
        idler_ks=1,
        label="Converter, ωI=ωp-ωs",
        save_name="continuous_vs_discrete_converter",
        backend=backend,
    )
