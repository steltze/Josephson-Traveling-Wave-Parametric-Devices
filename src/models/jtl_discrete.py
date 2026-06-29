from __future__ import annotations

import numpy as np

from models.cell import CellImmitance
from models.helpers import band_coupling


class JTLDiscrete:
    """
    Factory for Josephson Transmission Line unit cells.

    Right-handed (default): series inductor + shunt capacitor.
    Left-handed:            series capacitor + shunt inductor.

    In both topologies:
      - symmetric pi-section: end cells carry half the shunt element
      - first cell has no series element (open left port of the pi)
      - pump modulation acts on the Josephson (reactive) element:
          RH → Zs_harm  (series L modulated)
          LH → Yg_harm  (shunt L modulated)

    Usage
    -----
    sim = Simulation(JTL, cfg)                # right-handed
    sim = Simulation(JTL.left_handed(), cfg)  # left-handed
    """

    handedness: str = "right"

    @classmethod
    def left_handed(cls) -> type:
        """Return a JTL subclass configured as a left-handed transmission line."""

        class LHJTL(cls):
            handedness = "left"

        LHJTL.__name__ = "LHJTL"
        return LHJTL

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
        # element values — same parametrisation for both topologies:
        #   RH: L (series),  C (shunt)
        #   LH: C (series),  L (shunt)   ← same numbers, swapped roles
        """
        Add docs on values
        """
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

        lh = cls.handedness == "left"
        cells = []
        for i in range(ncell):
            first = i == 0

            if not lh:
                C_end = C[i] / (2.0 if (i == 0 or i == ncell - 1) else 1.0)
                _L, _wj, _eps, _th = L[i], wj[i], epsilons[i], thetas[i]
                _C = C_end

                n = len(config.ks_state)
                Zs_harm_arr = np.zeros((Nf, n, n), dtype=complex)
                if not first:
                    # Build modulated L matrix in sideband space: I + eps*B1 + eps^2*B2 + ...
                    L_mat = np.eye(n)
                    for p in range(1, M + 1):
                        L_mat += (_eps ** p) * band_coupling(n, p)
                    # Sideband frequencies for each signal freq: (Nf, n)
                    omega_sb = np.abs(w_s[:, None] + np.array(config.ks_state)[None, :] * w_p)
                    # (Nf, n, n) diagonal Omega matrices
                    Omega = np.zeros((Nf, n, n), dtype=float)
                    for _k in range(n):
                        Omega[:, _k, _k] = omega_sb[:, _k]
                    # Exact parallel-LC series impedance: JJ inductor || self-capacitance
                    Ccap_val = 1.0 / (_wj**2 * _L)
                    Yex = np.linalg.inv(1j * (Omega @ (_L * L_mat))) + 1j * Omega * Ccap_val
                    Zex = np.linalg.inv(Yex)  # (Nf, n, n)
                    # Subtract zero-order diagonal: Zs0[f,k] = j*omega_sb*L/(1-omega_sb^2/wj^2)
                    Zex_harm = Zex.copy()
                    Zex_harm[:, np.arange(n), np.arange(n)] -= (
                        1j * omega_sb * _L / (1.0 - omega_sb**2 / _wj**2)
                    )
                    # Full coupling matrix: Zs_harm_arr[f, target, source]
                    Zs_harm_arr = Zex_harm
                Yg_harm_arr = np.zeros((M, Nf), dtype=complex)

                cells.append(
                    CellImmitance(
                        theta=_th,
                        Zs0_fn=lambda w, L=_L, wji=_wj, f=first: (
                            0.0 if f else 1j * w * L / (1 - w**2 / wji**2)
                        ),
                        Yg0_fn=lambda w, C=_C: 1j * w * C,
                        Zs_harm_fn=Zs_harm_arr,
                        Yg_harm_fn=Yg_harm_arr,
                    )
                )
            else:
                L_end = L[i] / (2.0 if (i == 0 or i == ncell - 1) else 1.0)
                _C, _wj, _eps, _th = C[i], wj[i], epsilons[i], thetas[i]
                _L = L_end

                Zs_harm_arr = np.zeros((M, Nf), dtype=complex)
                Yg_harm_arr = np.zeros((M, Nf), dtype=complex)
                if M >= 1:
                    Yg_harm_arr[0] = 1j * _eps / (2 * w_s * _L)

                cells.append(
                    CellImmitance(
                        theta=_th,
                        Zs0_fn=lambda w, C=_C, f=first: 0.0 if f else 1 / (1j * w * C),
                        Yg0_fn=lambda w, L=_L, wji=_wj: (
                            (1 - w**2 / wji**2) / (1j * w * L)
                        ),
                        Zs_harm_fn=Zs_harm_arr,
                        Yg_harm_fn=Yg_harm_arr,
                    )
                )
        return cells
