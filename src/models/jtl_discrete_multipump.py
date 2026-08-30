from __future__ import annotations

import numpy as np

from models.cell import CellImmitance
from models.electrical_elements import (
    Capacitor,
    Component,
    ModulatedInductorMultiPump,
    multipump_frequency_grid,
    n_sidebands_from_Kmax,
)


class JTLDiscreteMultiPump:
    """
    Factory for Josephson Transmission Line unit cells modulated by P
    independent pump tones (see
    models.electrical_elements.ModulatedInductorMultiPump).

    Unlike single-pump JTLDiscrete, `config` carries per-pump sequences
    (all length P; P=2 for starters) in place of the scalar pump fields:
        omega_pump : sequence of P pump angular frequencies
        v_pump     : sequence of P pump phase velocities
        epsilon    : sequence of P modulation depths
        Kmax       : sequence of P (k_min, k_max) pairs, e.g.
                     [(-2, 3), (-1, 1)] (need not be symmetric about 0).
                     The tracked state is the FULL tensor lattice
                     n_sidebands[j] = k_max_j - k_min_j + 1,
                     D = prod(n_sidebands) (see multipump_frequency_grid
                     / n_sidebands_from_Kmax), replacing the single-pump
                     `ks_state` list.
    Everything else (Z0, ncell, cell_size, omega_cutoff, omega_j, nramp,
    disorder*, M, omegas) is shared/scalar exactly as in JTLDiscrete.
    """

    @classmethod
    def build(cls, config, cell_topology: str = "L") -> list[CellImmitance]:
        """
        Build one CellImmitance per unit cell.

        Parameters
        ----------
        config : SimulationConfig-like, with the per-pump fields above
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

        ncell = config.ncell

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

        w_p = list(config.omega_pump)  # P pump angular frequencies
        v_p = list(config.v_pump)  # P pump phase velocities
        P = len(w_p)
        if len(v_p) != P or len(config.epsilon) != P:
            raise ValueError("omega_pump, v_pump and epsilon must all have length P")

        epsilons = [profile * eps_j for eps_j in config.epsilon]  # P x (ncell,)
        wj = 1.0 / np.sqrt(L * Cs_jj)

        thetas = [w_p[j] / v_p[j] * ns * a for j in range(P)]  # P x (ncell,)

        # thetas = [np.sign(config.v_ratio[j])*dispersion_bloch(w_p[j], config.omega_cutoff/config.v_ratio[j])* ns for j in range(P)]  # P x (ncell,)

        Kmax = list(config.Kmax)
        n_sb = n_sidebands_from_Kmax(Kmax)

        M = config.M
        w_s = np.asarray(config.omegas)  # (Nf,)
        Nf = len(w_s)

        omega_sb = np.stack([multipump_frequency_grid(ws, w_p, Kmax)[0] for ws in w_s])
        n = omega_sb.shape[1]

        cells = []
        for i in range(ncell):
            first = cell_topology == "L" and i == 0

            halve_end = cell_topology == "L" and (i == 0 or i == ncell - 1)
            C_end = C[i] / (2.0 if halve_end else 1.0)
            _L, _wj = L[i], wj[i]
            _eps = [epsilons[j][i] for j in range(P)]
            _th = [thetas[j][i] for j in range(P)]
            _C = C_end

            Zs_harm_arr = np.zeros((Nf, n, n), dtype=complex)
            if not first:
                Ccap_val = 1.0 / (_wj**2 * _L)
                squid = Component.parallel(
                    ModulatedInductorMultiPump(
                        L0=_L, eps=_eps, n_sidebands=n_sb, order=M, theta=_th
                    ),
                    Capacitor(Ccap_val),
                )
                Zs_harm_arr = squid.impedance_matrix(omega_sb)  # (Nf, n, n)

            # Shunt-to-ground capacitor: unmodulated, so it doesn't couple
            # sidebands -- its harmonic matrix is diagonal.
            Yg_harm_arr = Capacitor(_C).admittance_matrix(omega_sb)  # (Nf, n, n)

            cells.append(
                CellImmitance(
                    Zs_harm_fn=Zs_harm_arr,
                    Yg_harm_fn=Yg_harm_arr,
                )
            )
        return cells
