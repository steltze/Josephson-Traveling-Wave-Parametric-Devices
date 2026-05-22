import numpy as np

from symbolic.cell_single_mode import CellSingleMode
from analysis.s_parameters import plot_s_parameters
from solver.abcd_matrix import ABCDMatrix
import matplotlib.pyplot as plt

def main():
    ## port impedance
    Z0 = 50

    ### signal frequency span
    freqs = np.linspace(0.5e9, 12e9, 10)

    ## add disorder to the values of L's and C's
    disorder = True
    disorderSpan = 0.01

    ## cell size (um) - should not impact final result
    a = 10e-6

    # number of cells
    ncell = 1
    ns = np.arange(ncell)
    M, ks_state = 1, [0, 1]
    epsilon = 0.2

    w_s = 50e9 * 2 * np.pi  # signal frequency
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
        print("!! w_p is negative !!")

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

    # ws = freqs * 2 * np.pi
    # wd = ws + w_p
    # freqsd = wd / 2 / np.pi

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

    cell = CellSingleMode()
    T_sym, state_syms, Zs_m, Yg_m = cell.build_symbolic_transfer_matrix(M, ks_state)
    dim = len(state_syms)

    freq_params_list = [_make_freq_params(f * 2 * np.pi, w_p) for f in freqs]

    cell_params = prepare_immitances(Css, Lss, Cgs, epsilonSs, w_s, thetas)
    T_grid = cell.build_cell_freq_matrices(
        T_sym, dim, M, ks_state, Zs_m, Yg_m, cell_params, freq_params_list
    )

    cascaded_S_matrix = ABCDMatrix.from_cell_grid(T_grid).to_S(Z0=Z0)
    print(cascaded_S_matrix.array)
    ax = plot_s_parameters(cascaded_S_matrix.array, freqs, [(1,1), (1,2), (2,1), (2, 2)])
    plt.show()
    
    return


def _make_freq_params(omega_s, omega_p):
    return dict(
        omega_val_fn=lambda k, os=omega_s: os + k * omega_p,
        omega_p_val=omega_p,
        k_val=0,
    )


def prepare_immitances(Cs, Ls, Cg, epsilonSs, w_s, thetas):
    N = len(Ls)
    w_j_s = 1 / np.sqrt(Ls * Cs)
    Zs0 = 1j * w_s * Ls / (1 - w_s**2 / w_j_s**2)
    Zs1 = Zs0 * epsilonSs / (1 - w_s**2 / w_j_s**2)  ### order 1 in epsilon
    Yg0 = 1j * w_s * Cg

    cell_params = [
        dict(
            Zs0_val=Zs0[i],
            Yg0_val=Yg0[i],
            theta_val=thetas[i],
            Zs_num_fn=lambda m, _, _z=Zs1[i]: {1: _z}.get(m, 0j),
            Yg_num_fn=lambda m, _: 0j,
        )
        for i in range(N)
    ]
    return cell_params

if __name__ == "__main__":
    main()
