import logging
import time
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt

from models.cell import CellImmitance
from symbolic.cell_single_mode import CellSingleMode
from analysis.s_parameters import plot_s_parameters
from solver.abcd_matrix import ABCDMatrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    log.info("[%s] %.4fs", label, time.perf_counter() - t0)


def main():
    Z0 = 50
    freqs = np.linspace(0.5e9, 12e9, 500)

    disorder = False
    disorder_span = 0.01
    seed = 42
    rng = np.random.default_rng(seed)

    # cell size (um) - does not impact final result
    a = 10e-6
    ncell = 500
    ns = np.arange(ncell)
    M, ks_state = 1, [0, 1]
    epsilon = 0.0
 
    # w_p  = 5e9 typical

    w_s = 50e9 * 2 * np.pi  # cutoff signal frequency
    w_j = 30e9 * 2 * np.pi  # junction plasma frequency
    w_ss = w_s * np.ones(ncell)
    w_js = w_j * np.ones(ncell)
    v_s = a * w_s

    w_c = 5e9 * 2 * np.pi  # ~center of the bandgap

    v_p = v_s / 3  # phase-matching condition

    # process phase-matched at w_c (continuous limit); all v's positive; assumes v_p < v_s, v_d
    k = 1
    w_p = v_s / v_p * w_c / k

    print(f"omega pump = {w_p/1e9} GHz")

    if w_p <= 0: ################## probably not correct
        raise ValueError("w_p is negative")

    # local pump velocity: conversion matched at w_c
    xmax, xmin = 0.95, 1.25
    vps = np.linspace(xmin * v_p, xmax * v_p, ncell)
    thetas = +w_p / vps * ns * a  # pump applied backward

    # adiabatic envelope (set nramp=0 to disable)
    nramp = 0
    if nramp > 0:
        alpha = 4 / nramp
        ramp_up = 0.5 * (1 + np.tanh(alpha * (ns - nramp / 2)))
        ramp_down = 0.5 * (1 + np.tanh(alpha * ((ncell - 1 - nramp / 2) - ns)))
        profile = ramp_up * ramp_down
    else:
        profile = np.ones(ncell)

    epsilonSs = profile * epsilon

    omegaRs = w_ss
    ZRs = Z0 * np.ones(ncell)
    Lss = ZRs / omegaRs
    Cgs = 1 / (omegaRs * ZRs)
    Css = 1 / (w_js**2 * Lss)

    if disorder:
        lo, hi = 1 - disorder_span / 2, 1 + disorder_span / 2
        Lss *= rng.uniform(lo, hi, ncell)
        Css *= rng.uniform(lo, hi, ncell)
        Cgs *= rng.uniform(lo, hi, ncell)

    solver = CellSingleMode()

    with timer("1 Symbolic transfer matrix"):
        T_sym, state_syms, Zs_m, Yg_m = solver.build_symbolic_transfer_matrix(
            M, ks_state
        )

    dim = len(state_syms)
    cells = prepare_immitances(Css, Lss, Cgs, epsilonSs, thetas)

    with timer("2 Numerical cell matrices"):
        T_grid = solver.build_cell_freq_matrices(
            T_sym,
            dim,
            M,
            ks_state,
            Zs_m,
            Yg_m,
            freqs * 2 * np.pi,
            w_p,
            cells,
        )

    with timer("3 S-matrix cascade"):
        cascaded_S_matrix = ABCDMatrix.from_cell_grid_S(T_grid, Z0=Z0)
    print(np.abs(cascaded_S_matrix.array)[:3])

    plot_s_parameters(
        cascaded_S_matrix.array,
        freqs,
        # [(2, 1)],
        [(3, 1), (1, 1)],
    )
    plt.show()


def prepare_immitances(Cs, Ls, Cg, epsilons, thetas) -> list[CellImmitance]:
    wj = 1 / np.sqrt(Ls * Cs)
    return [
        CellImmitance(
            theta=thetas[i],
            Zs0_fn=lambda w, L=Ls[i], wj_i=wj[i], i=i: (0.0 if i==0 else 1j * w * L),
            Yg0_fn=lambda w, C=Cg[i] / (2.0 if i in (0, len(Ls) - 1) else 1.0): 1j * w * C,
            Zs_harm_fn=lambda m, w, L=Ls[i], wj_i=wj[i], eps=epsilons[i], i=i: (
                1j * w * L * eps / ((1 - w**2 / wj_i**2) ** 2) if (m == 1 and i!=0) else 0.0
            ),
            Yg_harm_fn=lambda m, w: 0j * w,
        )
        for i in range(len(Ls))
    ]


def test_dispersion_relations(ks_state: list[int] | None = None):
    """
    Validate bloch_wavenumbers() for an NxN unit cell against the closed-form
    dispersion of two canonical ATL topologies.  Works for any ks_state.

    With w_p=0 the sidebands are decoupled, so all N/2 forward modes are
    degenerate and equal to the unperturbed single-mode analytical result:

      RH (series L, shunt C):  k_RH = arccos(1 - ω²LC/2)   in (0, π)
      LH (series C, shunt L):  k_LH = -arccos(1 - 1/(2ω²LC)) in (-π, 0)

    bloch_wavenumbers() sorts eigenvalues descending by k, so for any N:
      - index  0   → most positive k  (forward RH / backward LH)
      - index -1   → most negative k  (backward RH / forward LH)

    For dim=2 (ks_state=[0]) dispersion_relation() is also exercised as an
    independent cross-check of the half-trace formula (Eq. 24 of the paper).
    """
    if ks_state is None:
        ks_state = [0]

    L = 1e-9     # 1 nH
    C = 400e-15  # 400 fF

    wc_RH = 2.0 / np.sqrt(L * C)
    wc_LH = 1.0 / (2.0 * np.sqrt(L * C))

    M = 1
    freqs = np.linspace(0.0001e9, 22e9, 1000)
    w = 2.0 * np.pi * freqs

    print("=" * 60)
    print(f"ks_state={ks_state}  →  {2*len(ks_state)}x{2*len(ks_state)} T matrix")
    print(f"RH cutoff: {wc_RH / (2*np.pi) / 1e9:.2f} GHz  (k → π)")
    print(f"LH cutoff: {wc_LH / (2*np.pi) / 1e9:.2f} GHz  (k → −π)")

    # ------------------------------------------------------------------ #
    # Symbolic pipeline                                                    #
    # ------------------------------------------------------------------ #

    solver = CellSingleMode()

    with timer("Symbolic transfer matrix"):
        T_sym, state_syms, Zs_m, Yg_m = solver.build_symbolic_transfer_matrix(M, ks_state)
    dim = len(state_syms)

    cell_RH = [CellImmitance(
        theta=0.0,
        Zs0_fn=lambda w: 1j * w * L,
        Yg0_fn=lambda w: 1j * w * C,
        Zs_harm_fn=lambda _, w: 0j * w,
        Yg_harm_fn=lambda _, w: 0j * w,
    ) for i in range(1) ]
    cell_LH = [CellImmitance(
        theta=0.0,
        Zs0_fn=lambda w: 1.0 / (1j * w * C),
        Yg0_fn=lambda w: 1.0 / (1j * w * L),
        Zs_harm_fn=lambda _, w: 0j * w,
        Yg_harm_fn=lambda _, w: 0j * w,
    ) for i in range(1) ]

    with timer("Numerical cell matrices"):
        T_grid_RH = solver.build_cell_freq_matrices(
            T_sym, dim, M, ks_state, Zs_m, Yg_m, w, 0.0, cell_RH
        )
        T_grid_LH = solver.build_cell_freq_matrices(
            T_sym, dim, M, ks_state, Zs_m, Yg_m, w, 0.0, cell_LH
        )

    T_RH = ABCDMatrix(T_grid_RH)  # (Nf, dim, dim)
    T_RH = T_RH.cascade()
    T_LH = ABCDMatrix(T_grid_LH)
    T_LH = T_LH.cascade()
    # ------------------------------------------------------------------ #
    # Analytical dispersion  (NaN outside each passband)                  #
    # ------------------------------------------------------------------ #

    ht_RH = 1.0 - w**2 * L * C / 2.0
    ht_LH = 1.0 - 1.0 / (2.0 * w**2 * L * C)

    passband_RH = np.abs(ht_RH) <= 1.0
    passband_LH = np.abs(ht_LH) <= 1.0

    with np.errstate(invalid="ignore"):
        k_RH_ana = np.where(passband_RH, np.arccos(ht_RH), np.nan)
        k_LH_ana = np.where(passband_LH, -np.arccos(ht_LH), np.nan)

    # ------------------------------------------------------------------ #
    # Eigenvalue method — works for any N×N                               #
    #   sorted descending: index 0 = max k (fwd RH), index -1 = min k    #
    #   (fwd LH).  With w_p=0 all N/2 forward modes are degenerate.      #
    # ------------------------------------------------------------------ #

    _, k_RH_ev = T_RH.bloch_wavenumbers()  # (Nf, dim)
    _, k_LH_ev = T_LH.bloch_wavenumbers()

    print(k_RH_ev.shape)
    k_RH_ev_fwd = k_RH_ev[:, 0]    # most positive k  → forward RH
    k_RH_ev_bwd = k_RH_ev[:, -1]   # most negative k  → backward RH
    k_LH_ev_fwd = k_LH_ev[:, -1]   # most negative k  → forward LH
    k_LH_ev_bwd = k_LH_ev[:, 0]    # most positive k  → backward LH

    # ------------------------------------------------------------------ #
    # Half-trace cross-check — only valid for 2×2                         #
    # ------------------------------------------------------------------ #

    k_RH_ht = k_LH_ht = None
    if dim == 2:
        _, k_RH_ht = T_RH.dispersion_relation()
        _, k_LH_ht = T_LH.dispersion_relation()

    # ------------------------------------------------------------------ #
    # Assertions                                                           #
    # ------------------------------------------------------------------ #

    err_RH_ev = np.nanmax(np.abs(k_RH_ev_fwd[passband_RH] - k_RH_ana[passband_RH]))
    err_LH_ev = np.nanmax(np.abs(k_LH_ev_fwd[passband_LH] - k_LH_ana[passband_LH]))

    print(f"\nMax |Δk| vs analytical (rad/cell):")
    print(f"  RH  eigenvalue:  {err_RH_ev:.2e}   (expect < 1e-10)")
    print(f"  LH  eigenvalue:  {err_LH_ev:.2e}   (expect < 1e-10)")

    tol = 1e-10
    assert err_RH_ev < tol, f"RH eigenvalue FAILED: {err_RH_ev:.2e}"
    assert err_LH_ev < tol, f"LH eigenvalue FAILED: {err_LH_ev:.2e}"

    if dim == 2:
        err_RH_ht = np.nanmax(np.abs(k_RH_ht[passband_RH] - k_RH_ana[passband_RH]))
        err_LH_ht = np.nanmax(np.abs(k_LH_ht[passband_LH] - k_LH_ana[passband_LH]))
        print(f"  RH  half-trace:  {err_RH_ht:.2e}   (expect < 1e-10)")
        print(f"  LH  half-trace:  {err_LH_ht:.2e}   (expect < 1e-10)")
        assert err_RH_ht < tol, f"RH half-trace FAILED: {err_RH_ht:.2e}"
        assert err_LH_ht < tol, f"LH half-trace FAILED: {err_LH_ht:.2e}"

    print("\nAll assertions passed ✓")

    # ------------------------------------------------------------------ #
    # Plot                                                                 #
    # ------------------------------------------------------------------ #

    f_GHz = freqs / 1e9
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, title, k_ana, k_ev_fwd, k_ev_bwd, k_ht, wc, passband, ylim in [
        (
            axes[0], f"Right-handed ATL  ({dim}x{dim})",
            k_RH_ana, k_RH_ev_fwd, k_RH_ev_bwd, k_RH_ht,
            wc_RH, passband_RH, (-0.3, np.pi + 0.3),
        ),
        (
            axes[1], f"Left-handed ATL  ({dim}x{dim})",
            k_LH_ana, k_LH_ev_fwd, k_LH_ev_bwd, k_LH_ht,
            wc_LH, passband_LH, (-np.pi - 0.3, 0.3),
        ),
    ]:
        ax.plot(f_GHz, np.abs(k_ana), "r-", lw=2.5, label="analytical")
        # if k_ht is not None:
            # ax.plot(f_GHz, k_ht, "b--", lw=1.8, label="half-trace (Eq. 24)")
        ax.plot(f_GHz, np.abs(k_ev_fwd), "g:", lw=2.0, label="eigenvalue (forward)")
        # ax.plot(f_GHz, np.abs(k_ev_bwd), "m:", lw=1.2, alpha=0.5, label="eigenvalue (backward)")

        # ax.axvline(wc / (2*np.pi*1e9), color="k", ls="--", lw=1.0,
        #            label=f"ωc = {wc/(2*np.pi)/1e9:.2f} GHz")
        # ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        # ax.axhline(np.pi, color="gray", lw=0.5, ls=":", alpha=0.6)
        # ax.axhline(-np.pi, color="gray", lw=0.5, ls=":", alpha=0.6)

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("k (rad/cell)")
        ax.set_title(title)
        # ax.set_ylim(*ylim)
        # ax.set_xlim(f_GHz[0], f_GHz[-1])
        ax.legend(fontsize=8, loc="upper right" if "Right" in title else "lower right")
        ax.grid(True, alpha=0.25)

    # axes[0].annotate("k > 0\n(forward phase)", xy=(5, 1.2), fontsize=9, color="navy")
    # axes[1].annotate("k < 0\n(backward phase)", xy=(8, -1.8), fontsize=9, color="navy")

    plt.suptitle(
        f"Dispersion relation validation — RH and LH ATLs  (ks_state={ks_state})\n"
        r"Eigenvalue method: $\gamma_i = -\log(\lambda_i)$,  $k_i = \mathrm{Im}\{\gamma_i\}$",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # test_dispersion_relations(ks_state=[0, 1])
    main()
