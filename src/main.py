import numpy as np

from symbolic.cell_single_mode import CellSingleMode
from src.solver.s_matrix import ABCD_to_S


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
    epsilon = 0.2

    omegasigma0 = 50e9 * 2 * np.pi  # Σ mode cutoff
    omegadelta0 = 50e9 * 2 * np.pi * 1.0  # Δ mode cutoff
    omegaj0 = 30e9 * 2 * np.pi  # junction plasma frequency
    omegasigmas = omegasigma0 * np.ones(ncell)
    omegadeltas = omegadelta0 * np.ones(ncell)
    omegajs = omegaj0 * np.ones(ncell)
    v_sigma = a * omegasigma0
    v_delta = a * omegadelta0

    ### ~center of the bandgap
    w_c = 5e9 * 2 * np.pi

    ## set pump velocity at the ~center of gap
    v_p0 = v_sigma / 3  # phase-matching condition

    ###pump frequency.  process phase-matched at w_c, all v's are positive, assumes v_p<v_s,v_d
    k = 1 # omega_k = omega_s + k*omega_p
    w_p = (v_delta / v_sigma + 1) / (v_delta / v_p0 - 1) * w_c * 1/k

    if w_p < 0:
        print("!! w_p is negative !!")

    ### array defining local pump velocity within the line (center velocity : conversion matched at w_c)
    xmax = 0.95
    xmin = 1.25
    vps = np.linspace(xmin * v_p0, xmax * v_p0, ncell)
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

    ### translate above parameters to :
    # Cg: capa to ground
    # Ci: inner capa between the two lines. If omegasigmas=omegadeltas, should be 0, we should get simple case of a single line, simply Sigma/Delta ports are the same
    # Ls: series inductor (modulated)
    # Cs: series capacitor= junction capacitor
    omegaRs = omegasigmas
    ZRs = Z0 * np.ones(ncell) * 1
    Lss = ZRs / omegaRs
    Cgs = 1 / (omegaRs * ZRs)
    rspeed = omegadeltas**2 / omegasigmas**2
    # Cis = (1 - rspeed) / (2 * rspeed) * Cgs
    Css = 1 / (omegajs**2 * Lss)

    ws = freqs * 2 * np.pi
    wd = ws + w_p
    freqsd = wd / 2 / np.pi

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
