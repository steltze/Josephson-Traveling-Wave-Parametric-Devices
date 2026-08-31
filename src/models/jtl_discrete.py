from __future__ import annotations

import numpy as np

from models.cell import CellImmitance
from models.electrical_elements import Capacitor, Component, ModulatedInductor
from models.dispersion_relations import dispersion_bloch

class JTLDiscrete:
    """
    Factory for Josephson Transmission Line unit cells.
    """

    @classmethod
    def build(cls, config, cell_topology: str = "L") -> list[CellImmitance]:
        """
        Build one CellImmitance per unit cell.

        Parameters
        ----------
        config : SimulationConfig
        cell_topology : "L" or "pi"
            "L" (series-then-shunt) needs a half-shunt-capacitor boundary
            correction and a series-impedance-free first cell to terminate
            the ladder symmetrically (config.ncell = desired sections + 1).
            "pi" (shunt/2-series-shunt/2) already has a half-capacitor at
            each of its own two ends, so chaining config.ncell cells
            uniformly, with no boundary special-casing, already terminates
            correctly (config.ncell = desired sections).

        Returns
        -------
        list[CellImmitance], length config.ncell
        """
        v_p = config.v_pump
        ncell = config.ncell
        print(f"NUmber of cells {ncell}")

        if cell_topology == "L":
            ns = np.zeros(ncell)
            ns[1:] = np.arange(ncell - 1)
        else:
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
            frac = config.disorder_span
            lo = 1 - frac
            hi = 1 + frac
            L *= rng.uniform(lo, hi, ncell)
            C *= rng.uniform(lo, hi, ncell)
            Cs_jj *= rng.uniform(lo, hi, ncell)

        if config.epsilon_nramp > 0:
            alpha = 4.0 / config.epsilon_nramp
            ramp_up = 0.5 * (1 + np.tanh(alpha * (ns - config.epsilon_nramp / 2)))
            ramp_down = 0.5 * (
                1 + np.tanh(alpha * ((ncell - 1 - config.epsilon_nramp / 2) - ns))
            )
            profile = ramp_up * ramp_down
        else:
            profile = np.ones(ncell)
        
        wj = 1.0 / np.sqrt(L * Cs_jj)

        w_p = config.omega_pump
        # thetas = w_p / v_p * ns * a

        thetas = np.sign(config.v_ratio)*dispersion_bloch(w_p, config.omega_cutoff / config.v_ratio) * ns


        M = config.M
        w_s = np.asarray(config.omegas) 
        Nf = len(w_s)
        n = len(config.ks_state)

        from scipy.special import jv
        a = np.pi * config.phi_dc_frac
        b = np.pi * config.phi_rf_frac
        J = [jv(p, b) for p in range(M + 1)]

        prefactor = 2.0 / L[0]
        c = {}
        c[0] = prefactor * np.cos(a) * J[0]                    
        for m in range(1, M+1):
            if m % 2 == 0:   
                c[m] = prefactor * np.cos(a) * 2*(-1)**(m//2) * J[m]
            else:            
                c[m] = -prefactor * np.sin(a) * 2*(-1)**((m-1)//2) * J[m]

        cells = []
        for i in range(ncell):
            first = cell_topology == "L" and i == 0

            halve_end = cell_topology == "L" and (i == 0 or i == ncell - 1)
            C_end = C[i] / (2.0 if halve_end else 1.0)
            _L, _wj, _th = L[i], wj[i], thetas[i]
            _C = C_end

            # Sideband frequencies for each signal freq: (Nf, n)
            omega_sb = w_s[:, None] + np.array(config.ks_state)[None, :] * w_p

            coeffs_cell = {p: profile[i] * c[p] for p in range(1, M + 1)}
            coeffs_cell[0] = c[0]
            Zs_harm_arr = np.zeros((Nf, n, n), dtype=complex)
            if not first:
                Ccap_val = 1.0 / (_wj**2 * _L)
                if config.epsilon:
                    inductor = ModulatedInductor(
                        eps=profile[i] * config.epsilon, theta=_th, order=M, L0=_L
                    )
                else:
                    inductor = ModulatedInductor(coeffs=coeffs_cell, theta=_th, order=M)
                squid = Component.parallel(inductor, Capacitor(Ccap_val))
                Zs_harm_arr = squid.impedance_matrix(omega_sb)  # (Nf, n, n)

            # Shunt-to-ground capacitor: unmodulated, so it doesn't couple
            # sidebands -- its harmonic matrix is diagonal.
            Yg = Capacitor(_C)
            Yg_harm_arr = Yg.admittance_matrix(omega_sb)  # (Nf, n, n)

            cells.append(
                CellImmitance(
                    Zs_harm_fn=Zs_harm_arr,
                    Yg_harm_fn=Yg_harm_arr,
                )
            )
        return cells
