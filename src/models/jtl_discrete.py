from __future__ import annotations

import numpy as np

from models.cell import CellImmitance
from models.electrical_elements import ModulatedInductor, Capacitor, Parallel


class JTLDiscrete:
    """
    Factory for Josephson Transmission Line unit cells.
    """

    @classmethod
    def build(cls, config) -> list[CellImmitance]:
        """
        Build one CellImmitance per unit cell.

        Parameters
        ----------
        config : SimulationConfig

        Returns
        -------
        list[CellImmitance], length config.ncell
        """

        ncell = config.ncell
        ns = np.arange(ncell)
        a = config.cell_size

        ZR = config.Z0 * np.ones(ncell)
 
        L = ZR / config.omega_cutoff * 2
        C = 2 * 1.0 / (config.omega_cutoff * ZR)
        Cs_jj = 1.0 / (
            config.omega_j**2 * L
        )  # junction self-capacitance (RH) / series cap correction (LH)

        if config.disorder:
            rng = np.random.default_rng(config.disorder_seed)
            lo = 1 - config.disorder_span / 2
            hi = 1 + config.disorder_span / 2
            L *= rng.uniform(lo, hi, ncell)
            C *= rng.uniform(lo, hi, ncell)
            Cs_jj *= rng.uniform(lo, hi, ncell)

        if config.nramp > 0:
            alpha = 4.0 / config.nramp
            ramp_up = 0.5 * (1 + np.tanh(alpha * (ns - config.nramp / 2)))
            ramp_down = 0.5 * (
                1 + np.tanh(alpha * ((ncell - 1 - config.nramp / 2) - ns))
            )
            profile = ramp_up * ramp_down
        else:
            profile = np.ones(ncell)

        epsilons = profile * config.epsilon
        wj = 1.0 / np.sqrt(L * Cs_jj)

        w_p = config.omega_pump
        v_p = config.v_pump
        thetas = w_p / v_p * ns * a

        M = config.M
        w_s = np.asarray(config.omegas)   # (Nf,)
        Nf = len(w_s)

        cells = []
        for i in range(ncell):
            first = i == 0

            C_end = C[i] / (2.0 if (i == 0 or i == ncell - 1) else 1.0)
            _L, _wj, _eps, _th = L[i], wj[i], epsilons[i], thetas[i]
            _C = C_end

            n = len(config.ks_state)
            Zs_harm_arr = np.zeros((Nf, n, n), dtype=complex)
            if not first:
                # Sideband frequencies for each signal freq: (Nf, n)
                omega_sb = w_s[:, None] + np.array(config.ks_state)[None, :] * w_p
                # Exact parallel-LC series impedance: JJ inductor || self-capacitance
                Ccap_val = 1.0 / (_wj**2 * _L)
                Zs_element = Parallel(
                    ModulatedInductor(L0=_L, eps=_eps, order=M),
                    Capacitor(Ccap_val),
                )
                Zs_harm_arr = Zs_element.impedance(omega_sb)  # (Nf, n, n)
            Yg_harm_arr = np.zeros((M, Nf), dtype=complex)

            cells.append(
                CellImmitance(
                    theta=_th,
                    Zs0_fn=lambda w, L=_L, wji=_wj, f=first: 0.0,
                    Yg0_fn=lambda w, cap=Capacitor(_C): cap.admittance(w),
                    Zs_harm_fn=Zs_harm_arr,
                    Yg_harm_fn=Yg_harm_arr,
                )
            )
        return cells
