import numpy as np

from symbolic.cell_single_mode import CellSingleMode
from solver.ABCD import ABCD_to_S

_OMEGA_S = 1.0
_OMEGA_P = 0.1
_THETA = np.pi / 4
_ZS0 = 1.0 + 0.0j
_YG0 = 0.5 + 0.0j


def _omega_val(k):
    return _OMEGA_S + k * _OMEGA_P


def _Zs_num(m):
    return {1: 0.1 + 0.05j, 2: 0.02 + 0.01j}.get(m, 0 + 0j)


def _Yg_num(m):
    return {1: 0.05 + 0.02j, 2: 0.01 + 0.005j}.get(m, 0 + 0j)


def main():
    cell = CellSingleMode()
    M, ks_state = 1, [0, 1]

    T_sym, state_syms, Zs_m, Yg_m = cell.build_symbolic_transfer_matrix(M, ks_state)
    cell.export_matrix_plot(T_sym, state_syms, filename="transfer_matrix.pdf")

    dim = len(state_syms)

    T_num = cell.build_numeric_matrix(
        T_sym,
        dim,
        M,
        ks_state,
        Zs_m,
        Yg_m,
        _ZS0,
        _YG0,
        _THETA,
        _OMEGA_P,
        _omega_val,
        _Zs_num,
        _Yg_num,
        k_val=0,
    )
    S = ABCD_to_S(T_num, 50)
    print(S)


# ws = freqs * 2 * np.pi
# wd = ws + w_p
# freqsd = wd / 2 / np.pi


# out = Line(Lss, Css, Cgs, Cis, Z0, thetas, epsilonSs).forward(2 * np.pi * freqs)


# logS1S2 = 20 * np.log10(np.abs(out[:, 0, 2]))
# logS2S1 = 20 * np.log10(np.abs(out[:, 2, 0]))

# logD1D2 = 20 * np.log10(np.abs(out[:, 1, 3]))
# logD2D1 = 20 * np.log10(np.abs(out[:, 3, 1]))


# logS2D2 = 20 * np.log10(np.abs(out[:, 2, 3]))
# logD1S1 = 20 * np.log10(np.abs(out[:, 1, 0]))

# logD2S1 = 20 * np.log10(np.abs(out[:, 3, 0]))

# logS1S1 = 20 * np.log10(np.abs(out[:, 0, 0]))
# logD1D1 = 20 * np.log10(np.abs(out[:, 1, 1]))
# logD2S2 = 20 * np.log10(np.abs(out[:, 3, 2]))
# logS1D1 = 20 * np.log10(np.abs(out[:, 0, 1]))


# phaseD1S1 = np.angle((out[:, 1, 0]))
# phaseS1D1 = np.angle((out[:, 0, 1]))
# phaseS2S1 = np.angle((out[:, 2, 0]))
# phaseS1S2 = np.angle((out[:, 0, 2]))
# phaseS1S1 = np.angle((out[:, 0, 0]))

# plt.figure()
# plt.plot(freqs, logD1S1, label="|D1S1| (dB)")

# plt.plot(freqs, logS2S1, label="|S2S1| (dB)")
# plt.plot(freqs, logS1S1, label="|S1S1| (dB)")

# plt.plot(freqs, logD2S1, label="|D2S1| (dB)")
# plt.plot(freqs, logS2S1, label="|S2S1| (dB)")


# plt.ylim([-40, 1])


# plt.legend(loc="best")
# plt.show()


# plt.legend(loc="best")
# plt.show()


# if 1:
#     plt.figure()
#     if epsilon == 0:
#         plt.plot(freqs, phaseS1S2, label="arg(S1S2)")
#     else:
#         plt.plot(freqs, phaseD1S1, label="arg(D1S1)")
#     plt.legend(loc="best")
#     plt.show()


#     # 3 JJs in series
# Series(JJ(L), JJ(L), JJ(L))          # == JJ(3L)

# # 3 JJs in parallel
# Parallel(Series(JJ(L), JJ(L), JJ(L)), Capacitor(C_g))        # == JJ(L/3)

# # Two-mode Δ shunt
# Parallel(Capacitor(C_g), Capacitor(2*C_i))

# # SNAIL (future — just add a SNAIL class with phi^3 potential)
# Series(SNAIL(L_small, L_large, n=3, flux=0.5))

if __name__ == "__main__":
    main()
