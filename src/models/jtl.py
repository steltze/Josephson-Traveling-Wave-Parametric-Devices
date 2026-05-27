from __future__ import annotations

import numpy as np

from models.cell import CellImmitance


class JTL:
    """
    Factory for Josephson Transmission Line unit cells.

    Symmetric pi-topology:
      - first and last cells carry Cg/2 (half shunt capacitor)
      - first cell has no series inductance (open left end of pi section)

    Series impedance modulation:
      Zs(t) = Zs0(w) + epsilon * Zs0(w) / (1 - w^2/wj^2)^2 * 2*cos(wp*t + theta_n)

    Usage
    -----
    cells = JTL.build(config)
    """

    @staticmethod
    def build(config) -> list[CellImmitance]:
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
        Ls = ZR / config.omega_cutoff
        Cg = 1.0 / (config.omega_cutoff * ZR)
        Cs = 1.0 / (config.omega_j**2 * Ls)

        if config.disorder:
            rng = np.random.default_rng(config.disorder_seed)
            lo = 1 - config.disorder_span / 2
            hi = 1 + config.disorder_span / 2
            Ls *= rng.uniform(lo, hi, ncell)
            Cs *= rng.uniform(lo, hi, ncell)
            Cg *= rng.uniform(lo, hi, ncell)

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
        wj = 1.0 / np.sqrt(Ls * Cs)

        w_p = config.omega_pump
        v_p = config.v_pump
        thetas = w_p / v_p * ns * a

        cells = []
        for i in range(ncell):
            _first = i == 0
            _Cg = Cg[i] / (2.0 if (i == 0 or i == ncell - 1) else 1.0)
            _L, _wj, _eps, _th = Ls[i], wj[i], epsilons[i], thetas[i]

            cells.append(
                CellImmitance(
                    theta=_th,
                    Zs0_fn=lambda w, L=_L, first=_first: 0.0 if first else 1j * w * L,
                    Yg0_fn=lambda w, C=_Cg: 1j * w * C,
                    Zs_harm_fn=lambda m, w, L=_L, wji=_wj, eps=_eps, first=_first: (
                        1j * w * L * eps / (1 - w**2 / wji**2) ** 2
                        if (m == 1 and not first)
                        else 0.0
                    ),
                    Yg_harm_fn=lambda m, w: 0j,
                )
            )
        return cells
