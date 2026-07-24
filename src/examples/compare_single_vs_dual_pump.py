import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from simulation import SimulationConfig, Simulation
from models import JTLDiscrete
from models.jtl_discrete_multipump import JTLDiscreteMultiPump
from models.electrical_elements import multipump_frequency_grid
from analysis.checks import check_photon_flux_conservation
from examples.utils import save_all
from dashboard import Dashboard
from dashboard.dashboard import port_labels_from_ks_state, port_labels_from_multipump
from logger import get_logger, setup_logging

log = get_logger(__name__)


def single_vs_dual_pump(cell_topology: str = "pi", backend=None, dashboard=False):
    """
    Compare two different ways of driving the *same* 2nd-harmonic idler at
    omega_s + 2*omega_p:

      - single-pump, M=2 (models.jtl_discrete.JTLDiscrete): the Josephson
        nonlinearity gives a direct O(eps^2) coupling from signal to the
        k=+2 sideband *within one cell* (see the `order=M` term in
        ModulatedInductor / electrical_elements.py).
      - dual-pump, M=1, with both pump tones set to the SAME frequency
        (models.jtl_discrete_multipump.JTLDiscreteMultiPump): each pump
        only hops its own tensor index by one step per cell -- there is
        NO per-cell eps1*eps2 cross term (see ModulatedInductorMultiPump's
        docstring). The (k1=1, k2=1) idler, landing at the same
        omega_s + omega_p1 + omega_p2 = omega_s + 2*omega_p, can only
        appear after cascading many cells (a genuine two-hop process).

    Both dual-pump tones sit at the same fp as the single pump; the second
    tone's eps is deliberately weaker (eps**2) so its direct (k1=1,k2=0)/
    (k1=0,k2=1) single-hop sidebands don't swamp the (1,1) two-hop idler
    we're actually comparing against the single-pump curve.
    """
    fp = 6.8  # GHz, shared by the single pump and both dual-pump tones -- kept
    # off the freq_min/freq_max/n_freqs grid below so no k=-1 sideband lands
    # exactly on omega=0 (an exact hit makes that frequency's per-cell
    # impedance matrix singular: the sideband's whole row/col of the
    # modulated-inductor impedance vanishes)
    eps = 0.05

    common = dict(
        Z0=50,
        ncell=321,
        cell_size=10e-6,
        omega_cutoff=2 * 50 / 540e-3,
        omega_j=60 * 2 * np.pi,
        freq_min=1,
        freq_max=8,
        n_freqs=400,
    )

    cfg_single = SimulationConfig(
        M=2,
        ks_state=[0, 1, 2],
        omega_pump=fp * 2 * np.pi,
        epsilon=eps,
        v_ratio=-2.5,
        **common,
    )

    cfg_dual = SimulationConfig(
        M=2,  # single hop per pump per cell -- no per-cell eps1*eps2 term
        omega_pump=[fp * 2 * np.pi, fp * 2 * np.pi],
        epsilon=[eps, eps**2],
        v_ratio=[-2.5, -2.5],
        Kmax=[(0, 2), (0, 2)],
        **common,
    )

    sim_single = Simulation(JTLDiscrete, cfg_single, backend=backend, cell_topology=cell_topology)
    sim_dual = Simulation(JTLDiscreteMultiPump, cfg_dual, backend=backend, cell_topology=cell_topology)

    S_single = sim_single.get_s_matrix(normalize=True).array  # (Nf, 2n, 2n)
    S_dual = sim_dual.get_s_matrix(normalize=True).array      # (Nf, 2D, 2D)

    freqs = cfg_single.freqs  # cfg_dual shares the same freq_min/max/n_freqs

    # --- port bookkeeping ---
    n = len(cfg_single.ks_state)
    signal_single = cfg_single.ks_state.index(0)
    idler_single = cfg_single.ks_state.index(2)

    _, labels = multipump_frequency_grid(0.0, cfg_dual.omega_pump, cfg_dual.Kmax)
    D = len(labels)
    signal_dual = labels.index((0, 0))
    idler_dual = labels.index((1, 1))

    # Left signal port -> right (transmitted) port, in dB
    S_signal_single = 10 * np.log10(
        np.abs(S_single[:, n + signal_single, signal_single]) ** 2 + 1e-15
    )
    S_idler_single = 10 * np.log10(
        np.abs(S_single[:, n + idler_single, signal_single]) ** 2 + 1e-15
    )
    S_signal_dual = 10 * np.log10(
        np.abs(S_dual[:, D + signal_dual, signal_dual]) ** 2 + 1e-15
    )
    S_idler_dual = 10 * np.log10(
        np.abs(S_dual[:, D + idler_dual, signal_dual]) ** 2 + 1e-15
    )

    # --- photon-flux (Manley-Rowe) sanity check for both models ---
    check_single = check_photon_flux_conservation(
        S_single, cfg_single.omegas, cfg_single.omega_pump, cfg_single.ks_state
    )
    check_dual = check_photon_flux_conservation(
        S_dual, cfg_dual.omegas, cfg_dual.omega_pump, cfg_dual.Kmax
    )
    log.info(
        "photon-flux residual (signal port), max |.-eta|: single-pump=%.2e  dual-pump=%.2e",
        np.max(np.abs(check_single[:, signal_single] - 1.0)),
        np.max(np.abs(check_dual[:, signal_dual] - 1.0)),
    )

    # --- matplotlib: S-parameter overlay + signal-transmission error ---
    fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax.plot(freqs, S_signal_single, color="tab:blue", label=r"single-pump $M=2$: signal $\rightarrow$ signal")
    ax.plot(
        freqs, S_idler_single, color="tab:blue", ls="--",
        label=r"single-pump $M=2$: signal $\rightarrow$ idler ($2\omega_p$)",
    )
    ax.plot(freqs, S_signal_dual, color="tab:orange", label=r"dual-pump: signal $\rightarrow$ signal")
    ax.plot(
        freqs, S_idler_dual, color="tab:orange", ls="--",
        label=r"dual-pump: signal $\rightarrow$ idler ($\omega_{p1}+\omega_{p2}$)",
    )
    ax.set_ylabel(r"$|S|^2$ (dB)")
    ax.set_title(
        f"Single-pump (M=2) vs. degenerate dual-pump ($f_{{p1}}=f_{{p2}}={fp:.1f}$ GHz) "
        f"— {cell_topology} cell, N={cfg_single.ncell}, ε={eps}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    S_err = S_signal_single - S_signal_dual
    rms_err = np.sqrt(np.mean(S_err**2))
    ax_err.axhline(0, color="0.4", lw=1.0)
    ax_err.plot(freqs, S_err, color="#C0392B", lw=1.5, label=f"RMS error = {rms_err:.2f} dB")
    ax_err.set_xlabel("Signal frequency (GHz)")
    ax_err.set_ylabel(r"$S_{signal}$ error (dB)")
    ax_err.set_title(r"Signal transmission (left $\rightarrow$ right) error: single-pump $-$ dual-pump")
    ax_err.legend()
    ax_err.grid(True, alpha=0.3)

    plt.tight_layout()

    # save_all(prefix="compare_single_vs_dual_pump", fmt="svg")
    plt.show()


    if dashboard:
        Dashboard(
            [S_single, S_dual],
            freqs=freqs,
            labels=[f"single-pump (M=2, fp={fp:.1f} GHz)", f"dual-pump (fp1=fp2={fp:.1f} GHz)"],
            port_labels=[
                port_labels_from_ks_state(cfg_single.ks_state),
                port_labels_from_multipump(cfg_dual.omega_pump, cfg_dual.Kmax),
            ],
        ).run()


if __name__ == "__main__":
    setup_logging()
    single_vs_dual_pump(dashboard=True)
