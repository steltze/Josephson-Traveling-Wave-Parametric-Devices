import time

import numpy as np

from symbolic.cell_single_mode import CellImmitance, CellSingleMode
from analysis.s_parameters import plot_s_parameters
from solver.abcd_matrix import ABCDMatrix
import matplotlib.pyplot as plt


def main():
    ## port impedance
    Z0 = 50

    ### signal frequency span
    freqs = np.linspace(0.5e9, 12e9, 2000)

    ## add disorder to the values of L's and C's
    disorder = True
    disorderSpan = 0.01

    ## cell size (um) - should not impact final result
    a = 10e-6

    # number of cells
    ncell = 500
    ns = np.arange(ncell)
    M, ks_state = 1, [0, 1]
    epsilon = 0.0

    w_s = 50e9 * 2 * np.pi  # cutoff signal frequency
    w_j = 30e9 * 2 * np.pi  # junction plasma frequency
    w_ss = w_s * np.ones(ncell)
    w_js = w_j * np.ones(ncell)
    v_s = a * w_s

    ### ~center of the bandgap
    w_c = 5e9 * 2 * np.pi

    ## set pump velocity at the ~center of gap
    v_p = v_s / 3  # phase-matching condition

    ###pump frequency.  process phase-matched at w_c, all v's are positive, assumes v_p<v_s,v_d
    k = 1  # omega_k = omega_s + k*omega_p
    w_p = v_s / v_p * w_c * 1 / k

    if w_p < 0:
        raise ValueError("!! w_p is negative !!")

    ### array defining local pump velocity within the line (center velocity : conversion matched at w_c)
    xmax = 0.95
    xmin = 1.25
    vps = np.linspace(xmin * v_p, xmax * v_p, ncell)
    ## translate this in local pump phase
    thetas = +w_p / vps * ns * a  ## pump is applied backward

    ### adiabatic ramp up and ramp down of the modulation of L's (avoid ripple at gap edge). Set to 0 for no adiabatic ramp up
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
    ZRs = Z0 * np.ones(ncell) * 1
    Lss = ZRs / omegaRs
    Cgs = 1 / (omegaRs * ZRs)
    Css = 1 / (w_js**2 * Lss)

    if disorder:
        Lss *= np.random.uniform(
            1 - disorderSpan / 2, 1 + disorderSpan / 2, Lss.shape[0]
        )
        Css *= np.random.uniform(
            1 - disorderSpan / 2, 1 + disorderSpan / 2, Lss.shape[0]
        )
        Cgs *= np.random.uniform(
            1 - disorderSpan / 2, 1 + disorderSpan / 2, Lss.shape[0]
        )

    solver = CellSingleMode()

    start_time = time.time()
    T_sym, state_syms, Zs_m, Yg_m = solver.build_symbolic_transfer_matrix(M, ks_state)
    print(f"---[1] {(time.time() - start_time):0.4f} seconds ---")

    dim = len(state_syms)

    start_time = time.time()
    cells = prepare_immitances(Css, Lss, Cgs, epsilonSs, thetas)
    print(f"---[2] {(time.time() - start_time):0.4f} seconds ---")

    start_time = time.time()
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
    print(f"---[3] {(time.time() - start_time):0.4f} seconds ---")

    start_time = time.time()
    cascaded_S_matrix = ABCDMatrix.from_cell_grid_S(T_grid, Z0=Z0)
    ax = plot_s_parameters(
        cascaded_S_matrix.array, freqs, [(1, 1), (1, 2), (2, 1), (2, 2)]
    )
    print(f"---[4] {(time.time() - start_time):0.4f} seconds ---")
    # print(cascaded_S_matrix.array)
    plt.show()

    return


def prepare_immitances(Cs, Ls, Cg, epsilons, thetas) -> list[CellImmitance]:
    wj = 1 / np.sqrt(Ls * Cs)
    return [
        CellImmitance(
            theta=thetas[i],
            Zs0_fn=lambda w, L=Ls[i]: 1j * w * L,
            Yg0_fn=lambda w, C=Cg[i]: 1j * w * C,
            Zs_harm_fn=lambda m, w, L=Ls[i], wj_i=wj[i], eps=epsilons[i]: (
                1j * w * L * eps / (1 - w**2 / wj_i**2) if m == 1 else 0j * w
            ),
            Yg_harm_fn=lambda m, w: 0j * w,
        )
        for i in range(len(Ls))
    ]


if __name__ == "__main__":
    main()
