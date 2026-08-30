import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

from simulation import SimulationConfig, Simulation
from models.jtl_discrete_multipump import JTLDiscreteMultiPump
from models.electrical_elements import multipump_frequency_grid
from analysis.checks import check_photon_flux_conservation
from examples.utils import COLOR_JULIA, COLOR_PYTHON, PAPER_STYLE
from dashboard import Dashboard
from logger import get_logger, setup_logging

log = get_logger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")


def two_pump_jtl(cell_topology: str = "pi", backend=None, dashboard=False):
    """
    One JTL modulated by two independent pump tones at once (see
    models.jtl_discrete_multipump.JTLDiscreteMultiPump), each doing its
    own signal -> idler conversion:

        idler1 = signal + pump1
        idler2 = signal + pump2

    Kmax is a list of P (k_min, k_max) pairs, one per pump (see
    models.electrical_elements.ModulatedInductorMultiPump), giving a
    prod(k_max_j - k_min_j + 1)-state tensor lattice. Pairs need not be
    symmetric about 0.

    Uses the `Simulation` class with `cell_cls=JTLDiscreteMultiPump`.
    `Simulation.get_transfer_matrix_grid` only calls `cell_cls.build` +
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
        omega_pump=[10.0 * 2 * np.pi, 13.2 * 2 * np.pi],
        epsilon=[0.05, 0.07],
        v_ratio=[2.5, -6.0],
        Kmax=[(-1, 1), (-1, 1)],
        freq_min=1,
        freq_max=14,
        n_freqs=200,
    )
    log.info(
        "pump frequencies: %.3f GHz, %.3f GHz",
        cfg.omega_pump[0] / (2 * np.pi),
        cfg.omega_pump[1] / (2 * np.pi),
    )
    log.info(f"omega cutoff: {cfg.omega_cutoff / (2 * np.pi)}")

    sim = Simulation(
        JTLDiscreteMultiPump, cfg, backend=backend, cell_topology=cell_topology
    )
    S_total = sim.get_s_matrix(normalize=True).array  # (Nf, N, N)


    _, labels = multipump_frequency_grid(0.0, cfg.omega_pump, cfg.Kmax)
    D = len(labels)
    signal_idx = labels.index((0, 0))
    idler1_idx = labels.index((0, 1))
    # idler2_idx = labels.index((0, 1))
    # idler3_idx = labels.index((1, 1))

    freqs = cfg.freqs
    S31_signal = np.abs(S_total[:, D + signal_idx, signal_idx]) ** 2
    S_idler1 = np.abs(S_total[:, signal_idx, D + signal_idx]) ** 2
    # S_idler2 = np.abs(S_total[:, idler2_idx, signal_idx]) ** 2
    # S_idler3 = np.abs(S_total[:, idler3_idx, signal_idx]) ** 2

    fp1 = cfg.omega_pump[0] / (2 * np.pi)
    fp2 = cfg.omega_pump[1] / (2 * np.pi)

    # --- photon-flux (Manley-Rowe) conservation check ---
    # Sigma_i eta_i |S_ph[i,j]|^2 must equal eta_j = sign(omega_j) exactly
    # for a lossless-junction line, even with parametric gain -- see
    # analysis.checks.check_photon_flux_conservation. Port frequencies are
    # signed (an idler can sit at a negative omega_s + k.omega_p), so this
    # is a stronger, more physically meaningful check than plain unitarity.
    check = check_photon_flux_conservation(
        S_total, cfg.omegas, cfg.omega_pump, cfg.Kmax
    )  # (Nf, N) -- should sit near 1.0
    max_residual = np.abs(np.abs(check) - 1.0).max()
    log.info("Photon-flux conservation: max |residual - 1| = %.3e", max_residual)

    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.plot(
            freqs,
            10 * np.log10(S31_signal + 1e-15),
            label=r"signal left $\rightarrow$ signal right",
            color=COLOR_PYTHON, lw=1.5, solid_capstyle="round",
        )
        ax.plot(
            freqs,
            10 * np.log10(S_idler1 + 1e-15),
            label=r"signal right $\rightarrow$ signal left",
            color=COLOR_JULIA, lw=1.5, solid_capstyle="round",
        )
        # ax.plot(
        #     freqs,
        #     10 * np.log10(S_idler2 + 1e-15),
        #     label=rf"signal $\rightarrow$ idler$_2$ ($\omega_s+\omega_{{p2}}$, $f_{{p2}}={fp2:.2f}$ GHz)",
        # )
        # ax.plot(
        #     freqs,
        #     10 * np.log10(S_idler3 + 1e-15),
        #     label=rf"signal $\rightarrow$ idler$_3$ ($\omega_s+\omega_{{p1}}+\omega_{{p2}}$, $f_{{p1}}+f_{{p2}}={fp1 + fp2:.2f}$ GHz)",
        # )
        ax.set_ylabel(r"$|S|^2$ (dB)")
        ax.set_title(
            f"Two-pump JTL",
            loc="left",
        )
        ax.set_xlabel("Signal frequency (GHz)")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Figure-level legend above the panel -- keeps it clear of the data
        # instead of fighting for empty space inside the axes.
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.85))
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.56, 0.95),
            ncol=1, frameon=False, handlelength=2.6,
        )

        os.makedirs(FIGURES_DIR, exist_ok=True)
        svg_path = os.path.join(FIGURES_DIR, "multi_pump_jtl_example.svg")
        fig.savefig(svg_path)
        log.info("Saved %s", svg_path)
        png_path = os.path.join(FIGURES_DIR, "multi_pump_jtl_example.png")
        fig.savefig(png_path, dpi=300)
        log.info("Saved %s", png_path)

    plt.show()

    if dashboard:
        Dashboard(
            [S_total],
            freqs=freqs,
            labels=[f"two-pump JTL (ε={cfg.epsilon})"],
            omega_pump=cfg.omega_pump,
            Kmax=cfg.Kmax,
        ).run()


if __name__ == "__main__":
    setup_logging()
    two_pump_jtl()
