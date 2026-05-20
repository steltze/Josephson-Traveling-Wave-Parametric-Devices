# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:35:45 2020

@author: CCC
"""

import numpy as np
import matplotlib.pyplot as plt

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


## amplitude modulation for L's. !! convention: L(x,t)=L0*(1+2*epsilon*cos(wp*t))
epsilon = 0.2

## define local cutoff frequency / speed in the signal  line + plasma frequency of junctions. Here this is constant
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
w_p = (v_delta / v_sigma + 1) / (v_delta / v_p0 - 1) * w_c

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
Cis = (1 - rspeed) / (2 * rspeed) * Cgs
Css = 1 / (omegajs**2 * Lss)


### here defined for 4x4 matrix.
## In my calculation, I consider only excitation with Sigma symmetry at w_signal. It becomes coupled to Delta symmetry at w_idler -> 2 "frequency-symmetry" ports x 2 physical ports (left/right)
def redheffer_star(S2, S1):
    """
    Redheffer star product for cascading two 4-port S-matrices.

    Parameters
    ----------
    S1, S2 : ndarray, shape (Nf, 4, 4)
        Scattering matrices (frequency-dependent)

    Returns
    -------
    S : ndarray, shape (Nf, 4, 4)
        Cascaded scattering matrix
    """

    # Split into 2x2 blocks
    S1_11 = S1[:, :2, :2]
    S1_12 = S1[:, :2, 2:]
    S1_21 = S1[:, 2:, :2]
    S1_22 = S1[:, 2:, 2:]

    S2_11 = S2[:, :2, :2]
    S2_12 = S2[:, :2, 2:]
    S2_21 = S2[:, 2:, :2]
    S2_22 = S2[:, 2:, 2:]

    I = np.eye(2)[None, :, :]  # shape (1,2,2) for broadcasting

    # Inverses (core of the method)
    inv_1 = np.linalg.inv(I - S2_11 @ S1_22)
    inv_2 = np.linalg.inv(I - S1_22 @ S2_11)

    # Redheffer formulas
    S11 = S1_11 + S1_12 @ inv_1 @ S2_11 @ S1_21
    S12 = S1_12 @ inv_1 @ S2_12
    S21 = S2_21 @ inv_2 @ S1_21
    S22 = S2_22 + S2_21 @ inv_2 @ S1_22 @ S2_12

    # Reassemble
    S = np.zeros_like(S1, dtype=complex)
    S[:, :2, :2] = S11
    S[:, :2, 2:] = S12
    S[:, 2:, :2] = S21
    S[:, 2:, 2:] = S22

    return S


class ABCDmat(object):
    def __init__(
        self, Ls, Cs, Cg, Ci, Z0, thetas, epsilonSs, w_p=w_p, thetas_rev=None
    ):  ###HERE!!
        self.Cs = Cs
        self.Ls = Ls
        self.Cg = Cg
        self.Ci = Ci
        self.Z0 = Z0
        self.N = len(Cs)
        self.epsilonSs = epsilonSs
        self.w_p = w_p
        self.thetas = thetas
        self.thetas_rev = thetas_rev

    def forward(self, w):
        self.freq_len = len(w)

        Cs = self.Cs
        Ls = self.Ls
        Cg = self.Cg
        Ci = self.Ci

        w_p = self.w_p
        if self.thetas_rev is not None:
            eithetas = np.exp(1j * self.thetas) + np.exp(1j * self.thetas_rev)
        else:
            eithetas = np.exp(1j * self.thetas)
        epsilonSs = self.epsilonSs

        ### bad notations: wsigma and wdelta are signal and idler frequencies , s and g stand for series and ground
        wsigma = np.expand_dims(w, 1)
        wdelta = wsigma + w_p

        ws = 1 / np.sqrt(Ls * Cs)  # plasma frequency (of the in-series elements Cs, Ls)
        Zs0sigma = (
            1j * wsigma * Ls / (1 - wsigma**2 / ws**2)
        )  # divided by the correction
        Zs1sigma = (
            Zs0sigma * epsilonSs / (1 - wsigma**2 / ws**2)
        )  ### order 1 in epsilon
        Zs0delta = 1j * wdelta * Ls / (1 - wdelta**2 / ws**2)
        Zs1delta = (
            Zs0delta * epsilonSs / (1 - wdelta**2 / ws**2)
        )  ### order 1 in epsilon

        Yg0sigma = 1j * wsigma * Cg
        Yg1sigma = 0  # no off-diagonal terms, no pump modulcated corrections
        Yg0delta = 1j * wdelta * Cg + 1j * wdelta * 2 * Ci
        Yg1delta = 0

        Nf = len(wsigma)
        Nc = len(Ls)
        one = np.ones([Nf, Nc]) + 0j
        zero = np.zeros([Nf, Nc]) + 0j

        ### I MISTOOK CONVENTION: NORMAL MATRIX RELATES 1 TO 2, HERE 2 TO 1. I INVERT IT BELOW

        A = np.zeros((Nf, Nc, 2, 2), dtype=one.dtype)
        A[:, :, 0, 0] = one
        A[:, :, 0, 1] = zero
        A[:, :, 1, 0] = zero
        A[:, :, 1, 1] = one

        B = np.zeros((Nf, Nc, 2, 2), dtype=one.dtype)
        B[:, :, 0, 0] = -Zs0sigma
        B[:, :, 0, 1] = -Zs1delta * eithetas.conjugate()
        B[:, :, 1, 0] = -Zs1sigma * eithetas
        B[:, :, 1, 1] = -Zs0delta

        C = np.zeros((Nf, Nc, 2, 2), dtype=one.dtype)
        C[:, :, 0, 0] = -Yg0sigma
        C[:, :, 0, 1] = -Yg1delta * eithetas.conjugate()
        C[:, :, 1, 0] = -Yg1sigma * eithetas
        C[:, :, 1, 1] = -Yg0delta

        D = np.zeros((Nf, Nc, 2, 2), dtype=one.dtype)
        D[:, :, 0, 0] = 1 + Yg0sigma * Zs0sigma
        D[:, :, 0, 1] = (
            Zs1delta * Yg0sigma + Yg1delta * Zs0delta
        ) * eithetas.conjugate()
        D[:, :, 1, 0] = (Zs1sigma * Yg0delta + Yg1sigma * Zs0sigma) * eithetas
        D[:, :, 1, 1] = 1 + Yg0delta * Zs0delta

        mats = np.zeros((Nf, Nc, 4, 4), dtype=one.dtype)

        mats[:, :, :2, :2] = A
        mats[:, :, :2, 2:] = B
        mats[:, :, 2:, :2] = C
        mats[:, :, 2:, 2:] = D

        Nf, Nc = mats.shape[:2]

        ### Normalize S matrices to get a photon flux (instead of energy flux)
        sqrt_ws = np.sqrt(w)
        sqrt_wd = np.sqrt(w + w_p)

        D = np.zeros((len(w), 4, 4), dtype=complex)

        D[:, 0, 0] = sqrt_ws
        D[:, 1, 1] = sqrt_wd
        D[:, 2, 2] = sqrt_ws
        D[:, 3, 3] = sqrt_wd

        Dinv = np.zeros_like(D)
        Dinv[:, 0, 0] = 1 / sqrt_ws
        Dinv[:, 1, 1] = 1 / sqrt_wd
        Dinv[:, 2, 2] = 1 / sqrt_ws
        Dinv[:, 3, 3] = 1 / sqrt_wd
        D = D[:, None, :, :]  # (Nf, 1, 4, 4)
        Dinv = Dinv[:, None, :, :]
        converter = ABCD_to_S()

        mats_reshaped = mats.reshape(Nf * Nc, 4, 4)
        mats_reshaped = np.linalg.inv(mats_reshaped)

        Smats_reshaped = converter.forward(mats_reshaped, Z0)
        Smats = Smats_reshaped.reshape(Nf, Nc, 4, 4)
        Smats = D @ Smats @ Dinv

        Stot = Smats[:, 0]  # shape (Nf, 4, 4)

        for i in range(1, Smats.shape[1]):
            Stot = redheffer_star(Smats[:, i], Stot)
        return Stot


class ABCD_to_S(object):
    def forward(self, ABCD, Z0):
        self.ABCD = ABCD
        N = ABCD.shape[0]

        A = ABCD[:, :2, :2]
        B = ABCD[:, :2, 2:]
        C = ABCD[:, 2:, :2]
        D = ABCD[:, 2:, 2:]

        detC = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
        Cinv = np.empty((N, 2, 2), dtype=complex)
        Cinv[:, 0, 0] = C[:, 1, 1] / detC
        Cinv[:, 1, 1] = C[:, 0, 0] / detC
        Cinv[:, 0, 1] = -C[:, 0, 1] / detC
        Cinv[:, 1, 0] = -C[:, 1, 0] / detC

        Z = np.empty((N, 4, 4), dtype=complex)
        Z[:, :2, :2] = A @ Cinv
        Z[:, :2, 2:] = A @ Cinv @ D - B
        Z[:, 2:, :2] = Cinv
        Z[:, 2:, 2:] = Cinv @ D

        Zref = Z0 * np.tile(np.eye(4), (N, 1, 1))
        Gref = 1 / np.sqrt(Z0) * np.tile(np.eye(4), (N, 1, 1))

        S = Gref @ (Z - Zref) @ np.linalg.inv(Z + Zref) @ np.linalg.inv(Gref)
        return S


class Line(object):
    def __init__(self, Ls, Cs, Cg, Ci, Z0, thetas, epsilonSs, w_p=w_p, thetas_rev=None):
        self.Z0 = Z0
        self.N = len(Cs)
        self.ABCDmat_layer = ABCDmat(
            Ls, Cs, Cg, Ci, Z0, thetas, epsilonSs, w_p=w_p, thetas_rev=thetas_rev
        )
        self.ABCDtS = ABCD_to_S()

    def forward(self, w):
        out = self.ABCDmat_layer.forward(w)
        return out


if disorder:
    Lss *= np.random.uniform(1 - disorderSpan / 2, 1 + disorderSpan / 2, Lss.shape[0])
    Css *= np.random.uniform(1 - disorderSpan / 2, 1 + disorderSpan / 2, Lss.shape[0])
    Cgs *= np.random.uniform(1 - disorderSpan / 2, 1 + disorderSpan / 2, Lss.shape[0])


ws = freqs * 2 * np.pi
wd = ws + w_p
freqsd = wd / 2 / np.pi


out = Line(Lss, Css, Cgs, Cis, Z0, thetas, epsilonSs).forward(2 * np.pi * freqs)


logS1S2 = 20 * np.log10(np.abs(out[:, 0, 2]))
logS2S1 = 20 * np.log10(np.abs(out[:, 2, 0]))

logD1D2 = 20 * np.log10(np.abs(out[:, 1, 3]))
logD2D1 = 20 * np.log10(np.abs(out[:, 3, 1]))


logS2D2 = 20 * np.log10(np.abs(out[:, 2, 3]))
logD1S1 = 20 * np.log10(np.abs(out[:, 1, 0]))

logD2S1 = 20 * np.log10(np.abs(out[:, 3, 0]))

logS1S1 = 20 * np.log10(np.abs(out[:, 0, 0]))
logD1D1 = 20 * np.log10(np.abs(out[:, 1, 1]))
logD2S2 = 20 * np.log10(np.abs(out[:, 3, 2]))
logS1D1 = 20 * np.log10(np.abs(out[:, 0, 1]))


phaseD1S1 = np.angle((out[:, 1, 0]))
phaseS1D1 = np.angle((out[:, 0, 1]))
phaseS2S1 = np.angle((out[:, 2, 0]))
phaseS1S2 = np.angle((out[:, 0, 2]))
phaseS1S1 = np.angle((out[:, 0, 0]))

plt.figure()
plt.plot(freqs, logD1S1, label="|D1S1| (dB)")

plt.plot(freqs, logS2S1, label="|S2S1| (dB)")
plt.plot(freqs, logS1S1, label="|S1S1| (dB)")

plt.plot(freqs, logD2S1, label="|D2S1| (dB)")
plt.plot(freqs, logS2S1, label="|S2S1| (dB)")


plt.ylim([-40, 1])


plt.legend(loc="best")
plt.show()


plt.legend(loc="best")
plt.show()


if 1:
    plt.figure()
    if epsilon == 0:
        plt.plot(freqs, phaseS1S2, label="arg(S1S2)")
    else:
        plt.plot(freqs, phaseD1S1, label="arg(D1S1)")
    plt.legend(loc="best")
    plt.show()
