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

    disorder = True
    disorder_span = 0.01
    seed = 42
    rng = np.random.default_rng(seed)

    # cell size (um) - does not impact final result
    a = 10e-6
    ncell = 500
    ns = np.arange(ncell)
    M, ks_state = 1, [0]
    epsilon = 0.0

    w_s = 50e9 * 2 * np.pi  # cutoff signal frequency
    w_j = 30e9 * 2 * np.pi  # junction plasma frequency
    w_ss = w_s * np.ones(ncell)
    w_js = w_j * np.ones(ncell)
    v_s = a * w_s

    w_c = 5e9 * 2 * np.pi  # ~center of the bandgap

    v_p = v_s / 3  # phase-matching condition

    # process phase-matched at w_c; all v's positive; assumes v_p < v_s, v_d
    k = 1
    w_p = v_s / v_p * w_c / k

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
        # [(1, 1), (2, 2), (3, 1), (4, 2)],
        [(1, 1), (2, 1)],
    )
    plt.show()


def prepare_immitances(Cs, Ls, Cg, epsilons, thetas) -> list[CellImmitance]:
    wj = 1 / np.sqrt(Ls * Cs)
    return [
        CellImmitance(
            theta=thetas[i],
            Zs0_fn=lambda w, L=Ls[i], wj_i=wj[i]: 1j * w * L,
            Yg0_fn=lambda w, C=Cg[i]: 1j * w * C,
            Zs_harm_fn=lambda m, w, L=Ls[i], wj_i=wj[i], eps=epsilons[i]: (
                1j * w * L * eps / ((1 - w**2 / wj_i**2) ** 2) if m == 1 else 0j * w
            ),
            Yg_harm_fn=lambda m, w: 0j * w,
        )
        for i in range(len(Ls))
    ]


if __name__ == "__main__":
    main()
